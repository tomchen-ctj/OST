import numpy
import torch.distributed as dist
import torch
import clip
import os
import copy as cp


def reduce_tensor(tensor, n=None):
    if n is None:
        n = dist.get_world_size()
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt = rt / n
    return rt
   

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def sync(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        val = torch.tensor(self.val).cuda()
        sum_v = torch.tensor(self.sum).cuda()
        count = torch.tensor(self.count).cuda()
        self.val = reduce_tensor(val, world_size).item()
        self.sum = reduce_tensor(sum_v, 1).item()
        self.count = reduce_tensor(count, 1).item()
        self.avg = self.sum / self.count


def epoch_saving(config, epoch, model,  max_accuracy, optimizer, lr_scheduler, logger, working_dir, is_best):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    
    save_path = os.path.join(working_dir, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")
    if is_best:
        best_path = os.path.join(working_dir, f'best.pth')
        torch.save(save_state, best_path)
        logger.info(f"{best_path} saved !!!")


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    if os.path.isfile(config.MODEL.RESUME): 
        logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
        load_state_dict = checkpoint['model']

        # now remove the unwanted keys:
        if "module.prompt_learner.token_prefix" in load_state_dict:
            del load_state_dict["module.prompt_learner.token_prefix"]

        if "module.prompt_learner.token_suffix" in load_state_dict:
            del load_state_dict["module.prompt_learner.token_suffix"]

        if "module.prompt_learner.complete_text_embeddings" in load_state_dict:
            del load_state_dict["module.prompt_learner.complete_text_embeddings"]

        msg = model.load_state_dict(load_state_dict, strict=False)
        logger.info(f"resume model: {msg}")
        if config.TRAIN.LOAD_OPTIMIZER_STATE:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

                start_epoch = checkpoint['epoch'] + 1
                max_accuracy = checkpoint['max_accuracy']

                logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")

                del checkpoint
                torch.cuda.empty_cache()

                return start_epoch, max_accuracy
            except:
                del checkpoint
                torch.cuda.empty_cache()
                return 0, 0.
        else:
            logger.info(f'=> loaded model weights, but not optimizer state, from {config.MODEL.RESUME}')
            del checkpoint
            torch.cuda.empty_cache()
            return 0, 0.

    else:
        logger.info(("=> no checkpoint found at '{}'".format(config.MODEL.RESUME)))
        return 0, 0

def wise_state_dict(ori_model, loaded_state_dict, weight_for_origin = None):
    finetuned_model = cp.deepcopy( ori_model)
    msg = finetuned_model.load_state_dict(loaded_state_dict, strict = False)
    # todo   the  logit_scale does not affect the inference procedure
    print(f'load finetuned model {msg}')

    state_dict_ori = dict(ori_model.named_parameters())
    state_dict_finetuned = dict(finetuned_model.named_parameters())

    assert set(state_dict_ori) == set(state_dict_finetuned)

    fused_dict = {   k:  weight_for_origin * state_dict_ori[k]  + (1-weight_for_origin) * state_dict_finetuned[k] for k in state_dict_ori }
    return fused_dict
    # finetuned_model.load_state_dict( fused_dict )
    # return finetuned_model



def load_checkpoint_fuse(config, model, optimizer, lr_scheduler, logger):
    if os.path.isfile(config.MODEL.RESUME): 
        logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
        load_state_dict = checkpoint['model']

        # now remove the unwanted keys:
        if "module.prompt_learner.token_prefix" in load_state_dict:
            del load_state_dict["module.prompt_learner.token_prefix"]

        if "module.prompt_learner.token_suffix" in load_state_dict:
            del load_state_dict["module.prompt_learner.token_suffix"]

        if "module.prompt_learner.complete_text_embeddings" in load_state_dict:
            del load_state_dict["module.prompt_learner.complete_text_embeddings"]

        if config.TEST.ONLY_TEST and config.MODEL.FUSE_WEIGHT_FOR_ORIGIN != 0:
            # todo Wise FT, fuse the loaded model weights with original CLIP weights
            fused_state_dict = wise_state_dict(ori_model=model, loaded_state_dict=load_state_dict, weight_for_origin=config.MODEL.FUSE_WEIGHT_FOR_ORIGIN)
            msg = model.load_state_dict(fused_state_dict, strict = False)
            logger.info(f"Wise FT weight for origin {config.MODEL.FUSE_WEIGHT_FOR_ORIGIN}, fused model {msg}")
        else:
            msg = model.load_state_dict(load_state_dict, strict=False)
            logger.info(f"resume model: {msg}")

        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            start_epoch = checkpoint['epoch'] + 1
            max_accuracy = checkpoint['max_accuracy']

            logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
            
            del checkpoint
            torch.cuda.empty_cache()

            return start_epoch, max_accuracy
        except:
            del checkpoint
            torch.cuda.empty_cache()
            return 0, 0.

    else:
        logger.info(("=> no checkpoint found at '{}'".format(config.MODEL.RESUME)))
        return 0, 0


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def generate_text(data):
    text_aug = f"{{}}"
    classes = torch.cat([clip.tokenize(text_aug.format(c), context_length=77) for i, c in data.classes])

    return classes

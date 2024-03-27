import torch
import clip

def text_prompt(data):
    # text_aug = ['{}']
    text_aug = ['a video of a person {}.']
    text_dict = {}
    num_text_aug = len(text_aug)

    for ii, txt in enumerate(text_aug):
        text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for i, c in data.classes])

    classes = text_dict[0] 

    return classes.cuda()


def spatial_prompt(data):
    # text_aug = 'A video of {} usually includes {}'
    text_aug = 'This is a video about {}'
    # print("spatial prompt only")
    classes = []
    for key, category_name in zip(data.description_spatial, data.classes):
        buffer = []
        for capt in data.description_spatial[key]:
            # buffer.append(clip.tokenize(text_aug.format(category_name[1], capt)))
            buffer.append(clip.tokenize(text_aug.format(capt)))
        classes.append(torch.stack(buffer).squeeze()) # 8 77
    classes = torch.stack(classes)

    return classes.cuda()

def temporal_prompt(data):
    text_aug = 'A video of {} usually includes {}'
    # text_aug = 'This is a video about {}'
    classes = []
    for key, category_name in zip(data.description_spatial, data.classes):
        buffer = []
        for capt in data.description_temporal[key]:
            buffer.append(clip.tokenize(text_aug.format(category_name[1], capt)))
            # buffer.append(clip.tokenize(text_aug.format(capt)))
        classes.append(torch.stack(buffer).squeeze()) # 8 77
    classes = torch.stack(classes)

    return classes.cuda()
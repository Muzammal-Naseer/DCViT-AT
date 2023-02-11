import torch
from pathways.utils import get_model

# GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_resolution_ensemble(args):
    # We assume there are four models available at resolution: 56, 96, 120, 224
    # We load all of these models
    src_model_56, src_mean, src_std = get_model(args.src_model, args.num_classes, args)
    checkpoint = torch.load(f"pretrained_models/{args.src_model}_56/checkpoint.pth")
    src_model_56.load_state_dict(checkpoint['model'])
    src_model_56 = src_model_56.to(device).eval()

    src_model_96, _, _ = get_model(args.src_model, args.num_classes, args)
    checkpoint = torch.load(f"pretrained_models/{args.src_model}_96/checkpoint.pth")
    src_model_96.load_state_dict(checkpoint['model'])
    src_model_96 = src_model_96.to(device).eval()

    src_model_120, _, _ = get_model(args.src_model, args.num_classes, args)
    checkpoint = torch.load(f"pretrained_models/{args.src_model}_120/checkpoint.pth")
    src_model_120.load_state_dict(checkpoint['model'])
    src_model_120 = src_model_120.to(device).eval()

    src_model_224, _, _ = get_model(args.src_model[0:-8], args.num_classes, args)
    src_model_224 = src_model_224.to(device).eval()

    res_list = args.res.split('_')

    length = len(res_list)
    if length == 1 and args.res=='56':
        def forward_pass(image, return_tokens=False):
            output = src_model_56(image, return_tokens=return_tokens)
            return output
    elif length == 1 and args.res=='96':
        def forward_pass(image, return_tokens=False):
            output = src_model_96(image, return_tokens=return_tokens)
            return output
    elif length == 1 and args.res=='120':
        def forward_pass(image, return_tokens=False):
            output = src_model_120(image, return_tokens=return_tokens)
            return output
    elif length == 1 and args.res=='224':
        def forward_pass(image, return_tokens=False):
            output = src_model_224(image, return_tokens=return_tokens)
            return output
    elif length > 1 and args.res =='56_96':
        def forward_pass(image, return_tokens=False):
            output1 = src_model_56(image, return_tokens=return_tokens)
            output2 = src_model_96(image, return_tokens=return_tokens)
            if return_tokens:
                # Model returns tuple of Task specific cls token in logit space
                # and Task independent cls token in feature space
                return [x + y  for x, y in zip(output1[0], output2[0])], \
                       [x + y  for x, y in zip(output1[1], output2[1])]
            else:
                return [x + y  for x, y in zip(output1, output2)]
    elif length > 1 and args.res == '56_96_120':
        def forward_pass(image, return_tokens=False):
            output1 = src_model_56(image, return_tokens=return_tokens)
            output2 = src_model_96(image, return_tokens=return_tokens)
            output3 = src_model_120(image, return_tokens=return_tokens)
            if return_tokens:
                # Model returns tuple of Task specific cls token in logit space
                # and Task independent cls token in feature space
                return [x + y + z for x, y, z in zip(output1[0], output2[0], output3[0])], \
                       [x + y + z for x, y, z in zip(output1[1], output2[1], output3[1])]
            else:
                return [x + y + z for x, y, z in zip(output1, output2, output3)]
    elif length > 1 and args.res == '56_96_120_224':
        def forward_pass(image, return_tokens=False):
            output1 = src_model_56(image, return_tokens=return_tokens)
            output2 = src_model_96(image, return_tokens=return_tokens)
            output3 = src_model_120(image, return_tokens=return_tokens)
            output4 = src_model_224(image, return_tokens=return_tokens)
            if return_tokens:
                # Model returns tuple of Task specific cls token in logit space
                # and Task independent cls token in feature space
                return [x + y + z + q for x, y, z, q in zip(output1[0], output2[0], output3[0], output4[0])], \
                       [x + y + z + q for x, y, z, q in zip(output1[1], output2[1], output3[1], output4[1])]
            else:
                return [x + y + z + q for x, y, z, q in zip(output1, output2, output3, output4)]

    return forward_pass, src_mean, src_std
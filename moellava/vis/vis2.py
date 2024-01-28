import argparse
import torch
from tqdm import tqdm
from torch.nn import functional as F
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

def draw(args):
    data = torch.load(args.input)
    all_text_expert_counter_list = []
    all_img_expert_counter_list = []
    for k, v in tqdm(data.items()):
        gating_logit = v['gating_logit']
        images = v['images'][0] if v['images'] is not None else v['images']
        input_ids = v['input_ids'][0].tolist()
        output_ids = v['output_ids'][0].tolist()
        gating_logit = v['gating_logit']
        num_moe_layers = len(gating_logit)

        if images is not None:
            assert gating_logit[0].shape[0] + 1 == len(output_ids) + 575
            img_idx = output_ids.index(-200)
            output_ids = output_ids[:img_idx] + [-200] * 576 + output_ids[img_idx+1:]

            text_expert_counters = []
            img_expert_counters = []
            for layer_idx, logits in enumerate(gating_logit):

                assert logits.shape[0] + 1 == len(output_ids)  # double check
                num_expert = logits.shape[1]
                gates = F.softmax(logits, dim=1)
                indices1_s = torch.argmax(gates, dim=1)  # Create a mask for 1st's expert per token
                mask1 = F.one_hot(indices1_s, num_classes=int(gates.shape[1]))
                exp_counts = torch.sum(mask1, dim=0).detach().to('cpu')  # gating decisions

                text_indices1_s = torch.cat([indices1_s[:img_idx], indices1_s[img_idx+576+1:]])
                img_indices1_s = indices1_s[img_idx:img_idx+576]
                text_expert_counter = Counter(text_indices1_s.tolist())
                img_expert_counter = Counter(img_indices1_s.tolist())


                text_expert_counter_list = [text_expert_counter[k] for k in range(num_expert)]
                img_expert_counter_list = [img_expert_counter[k] for k in range(num_expert)]


                text_expert_counters.append(text_expert_counter_list)
                img_expert_counters.append(img_expert_counter_list)
            all_text_expert_counter_list.append(text_expert_counters)
            all_img_expert_counter_list.append(img_expert_counters)
    print()

    all_text_expert_counter_list = np.array(all_text_expert_counter_list)
    all_img_expert_counter_list = np.array(all_img_expert_counter_list)

    all_text_expert_counter = np.sum(all_text_expert_counter_list, axis=-1, keepdims=True)
    all_text_expert_counter = np.mean(all_text_expert_counter_list / all_text_expert_counter, axis=0)

    all_img_expert_counter = np.sum(all_img_expert_counter_list, axis=-1, keepdims=True)
    all_img_expert_counter = np.mean(all_img_expert_counter_list / all_img_expert_counter, axis=0)



    num_layer = all_text_expert_counter.shape[0]
    categories = [i*2+1 for i in range(num_layer)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

    bar_positions = np.arange(len(categories))

    colors = ['#62A0CA', '#FFA556', '#6BBC6B', '#E26868']
    ax1.bar(bar_positions, all_text_expert_counter[:, 0], color=colors[0], label='Expert 1')
    for i in range(1, num_expert):
        ax1.bar(bar_positions, all_text_expert_counter[:, i], bottom=np.sum(all_text_expert_counter[:, :i], axis=1), color=colors[i], label=f'Expert {i+1}')

    ax2.bar(bar_positions, all_img_expert_counter[:, 0], color=colors[0], label='Expert 1')
    for i in range(1, num_expert):
        ax2.bar(bar_positions, all_img_expert_counter[:, i], bottom=np.sum(all_img_expert_counter[:, :i], axis=1), color=colors[i], label=f'Expert {i+1}')

    # 设置x轴标签、标题和图例
    ax1.set_xlabel('MoE layer idx')
    ax1.set_ylabel('Percentage')
    ax1.set_xticks(bar_positions)
    ax1.set_xticklabels(categories)
    ax1.legend(loc='upper center', ncol=2)
    ax1.set_title('Text')
    ax1.set_ylim(0, 1.25)
    ax1.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax1.axhline(y=0.25, color='gray', linestyle='--')
    ax1.axhline(y=0.5, color='gray', linestyle='--')
    ax1.axhline(y=0.75, color='gray', linestyle='--')

    # 设置x轴标签、标题和图例
    ax2.set_xlabel('MoE layer idx')
    # ax2.set_ylabel('Percentage')
    ax2.set_xticks(bar_positions)
    ax2.set_xticklabels(categories)
    ax2.legend(loc='upper center', ncol=2)
    ax2.set_title('Image')
    ax2.set_ylim(0, 1.25)
    ax2.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax2.axhline(y=0.25, color='gray', linestyle='--')
    ax2.axhline(y=0.5, color='gray', linestyle='--')
    ax2.axhline(y=0.75, color='gray', linestyle='--')

    # 显示图形
    plt.tight_layout()
    if args.output is not None:
        plt.savefig(args.output)
    else:
        plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default='phi_sciqa.pt')
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    draw(args)

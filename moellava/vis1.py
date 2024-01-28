import torch
from tqdm import tqdm
from torch.nn import functional as F
from collections import Counter
import numpy as np
import argparse
import matplotlib.pyplot as plt


def draw(args):
    data = torch.load(args.input)
    all_text_img_expert_counter_list = []
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

            text_img_expert_counters = []
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

                len_text = len(text_indices1_s)
                len_img = len(img_indices1_s)
                scale = len_img / len_text

                text_img_expert_counter_list = [[int(text_expert_counter[k] * scale), img_expert_counter[k]] for k in range(num_expert)]

                text_img_expert_counters.append(text_img_expert_counter_list)
            all_text_img_expert_counter_list.append(text_img_expert_counters)
    print()

    all_text_img_expert_counter_list = np.array(all_text_img_expert_counter_list)
    all_text_img_expert_counter = all_text_img_expert_counter_list / np.sum(all_text_img_expert_counter_list, axis=-1, keepdims=True)
    all_text_img_expert_counter = np.mean(all_text_img_expert_counter, axis=0)

    all_text_img_expert = np.sum(all_text_img_expert_counter_list, axis=-1)
    all_text_img_expert = all_text_img_expert / np.sum(all_text_img_expert, axis=-1, keepdims=True)
    all_text_img_expert = np.mean(all_text_img_expert, axis=0)

    num_layer = all_text_img_expert_counter.shape[0]
    categories = [i*2+1 for i in range(num_layer)]
    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(1, 5, figsize=(20, 4), sharey=True)

    bar_positions = np.arange(len(categories))
    colors = ['#62A0CA', '#FFA556', '#6BBC6B', '#E26868']
    colors1 = ['#7CABCA', '#4996CA']
    colors2 = ['#FFBC80', '#FF9A40']
    colors3 = ['#7FBC7F', '#508D50']
    colors4 = ['#E28787', '#E24E4E']

    ax0.bar(bar_positions, all_text_img_expert[:, 0], color=colors[0], label='Expert 1')
    ax0.bar(bar_positions, all_text_img_expert[:, 1], color=colors[1], bottom=all_text_img_expert[:, 0], label='Expert 2')
    ax0.bar(bar_positions, all_text_img_expert[:, 2], color=colors[2], bottom=all_text_img_expert[:, 0]+all_text_img_expert[:, 1], label='Expert 3')
    ax0.bar(bar_positions, all_text_img_expert[:, 3], color=colors[3], bottom=all_text_img_expert[:, 0]+all_text_img_expert[:, 1]+all_text_img_expert[:, 2], label='Expert 4')


    ax1.bar(bar_positions, all_text_img_expert_counter[:, 0, 0], color=colors1[0], label='Text')
    ax1.bar(bar_positions, all_text_img_expert_counter[:, 0, 1], color=colors1[1], bottom=all_text_img_expert_counter[:, 0, 0], label='Image')

    ax2.bar(bar_positions, all_text_img_expert_counter[:, 1, 0], color=colors2[0], label='Text')
    ax2.bar(bar_positions, all_text_img_expert_counter[:, 1, 1], color=colors2[1], bottom=all_text_img_expert_counter[:, 1, 0], label='Image')

    ax3.bar(bar_positions, all_text_img_expert_counter[:, 2, 0], color=colors3[0], label='Text')
    ax3.bar(bar_positions, all_text_img_expert_counter[:, 2, 1], color=colors3[1], bottom=all_text_img_expert_counter[:, 2, 0], label='Image')

    ax4.bar(bar_positions, all_text_img_expert_counter[:, 3, 0], color=colors4[0], label='Text')
    ax4.bar(bar_positions, all_text_img_expert_counter[:, 3, 1], color=colors4[1], bottom=all_text_img_expert_counter[:, 3, 0], label='Image')


    ax0.set_xlabel('MoE layer')
    ax0.set_ylabel('Percentage')
    ax0.set_xticks(bar_positions)
    ax0.set_xticklabels(categories)
    # ax0.legend(loc='upper center', ncol=4)
    ax0.legend(loc='upper center', ncol=2)
    ax0.set_title('All experts')
    ax0.set_ylim(0, 1.25)
    ax0.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax0.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax0.axhline(y=0.25, color='gray', linestyle='--')
    ax0.axhline(y=0.5, color='gray', linestyle='--')
    ax0.axhline(y=0.75, color='gray', linestyle='--')

    ax1.set_xlabel('MoE layer')
    # ax1.set_ylabel('Percentage')
    ax1.set_xticks(bar_positions)
    ax1.set_xticklabels(categories)
    ax1.legend(loc=(0.24, 0.85), ncol=2)
    ax1.set_title('Expert 1')
    ax1.set_ylim(0, 1.25)
    ax1.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    # ax1.axhline(y=0.25, color='gray', linestyle='--')
    ax1.axhline(y=0.5, color='gray', linestyle='--')
    # ax1.axhline(y=0.75, color='gray', linestyle='--')

    ax2.set_xlabel('MoE layer')
    # ax2.set_ylabel('Percentage')
    ax2.set_xticks(bar_positions)
    ax2.set_xticklabels(categories)
    ax2.legend(loc=(0.24, 0.85), ncol=2)
    ax2.set_title('Expert 2')
    ax2.set_ylim(0, 1.25)
    ax2.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    # ax2.axhline(y=0.25, color='gray', linestyle='--')
    ax2.axhline(y=0.5, color='gray', linestyle='--')
    # ax2.axhline(y=0.75, color='gray', linestyle='--')

    ax3.set_xlabel('MoE layer')
    # ax3.set_ylabel('Percentage')
    ax3.set_xticks(bar_positions)
    ax3.set_xticklabels(categories)
    ax3.legend(loc=(0.24, 0.85), ncol=2)
    ax3.set_title('Expert 3')
    ax3.set_ylim(0, 1.25)
    ax3.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax3.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    # ax3.axhline(y=0.25, color='gray', linestyle='--')
    ax3.axhline(y=0.5, color='gray', linestyle='--')
    # ax3.axhline(y=0.75, color='gray', linestyle='--')

    ax4.set_xlabel('MoE layer')
    # ax4.set_ylabel('Percentage')
    ax4.set_xticks(bar_positions)
    ax4.set_xticklabels(categories)
    ax4.legend(loc=(0.24, 0.85), ncol=2)
    ax4.set_title('Expert 4')
    ax4.set_ylim(0, 1.25)
    ax4.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax4.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    # ax4.axhline(y=0.25, color='gray', linestyle='--')
    ax4.axhline(y=0.5, color='gray', linestyle='--')
    # ax4.axhline(y=0.75, color='gray', linestyle='--')

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

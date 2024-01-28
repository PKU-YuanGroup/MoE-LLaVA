import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mplsoccer import Bumpy, FontManager, add_image
import torch
from tqdm import tqdm
from torch.nn import functional as F


def draw(args):
    data = torch.load(args.input)
    all_text_token_path = []
    all_img_token_path = []
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

            text_token_path = []
            img_token_path = []
            for layer_idx, logits in enumerate(gating_logit):

                assert logits.shape[0] + 1 == len(output_ids)  # double check
                num_expert = logits.shape[1]
                gates = F.softmax(logits, dim=1)
                indices1_s = torch.argmax(gates, dim=1)  # Create a mask for 1st's expert per token
                mask1 = F.one_hot(indices1_s, num_classes=int(gates.shape[1]))
                exp_counts = torch.sum(mask1, dim=0).detach().to('cpu')  # gating decisions

                text_indices1_s = torch.cat([indices1_s[:img_idx], indices1_s[img_idx+576+1:]])
                img_indices1_s = indices1_s[img_idx:img_idx+576]

                text_token_path.append(text_indices1_s)
                img_token_path.append(img_indices1_s)
            text_token_path = torch.stack(text_token_path).T  # 每个token沿层的路径
            img_token_path = torch.stack(img_token_path).T  # 每个token沿层的路径

            all_text_token_path.append(text_token_path)
            all_img_token_path.append(img_token_path)
    print()

    all_text_token_path = torch.cat(all_text_token_path, dim=0)
    all_img_token_path = torch.cat(all_img_token_path, dim=0)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=12)
    new_all_text_token_path = pca.fit_transform(all_text_token_path.T)
    new_all_text_token_path = new_all_text_token_path.T
    new_all_text_token_path = ((new_all_text_token_path - np.min(new_all_text_token_path, keepdims=True, axis=1)) / (np.max(new_all_text_token_path, keepdims=True, axis=1) - np.min(new_all_text_token_path, keepdims=True, axis=1)))
    new_all_text_token_path = np.clip(new_all_text_token_path, a_min=0.01, a_max=0.99)
    new_all_text_token_path = {'text_'+str(n+1): [int(i // (1/num_expert) + 1) for i in j] for n, j in enumerate(new_all_text_token_path)}


    from sklearn.decomposition import PCA
    pca = PCA(n_components=12)
    new_all_img_token_path = pca.fit_transform(all_img_token_path.T)
    new_all_img_token_path = new_all_img_token_path.T
    new_all_img_token_path = ((new_all_img_token_path - np.min(new_all_img_token_path, keepdims=True, axis=1)) / (np.max(new_all_img_token_path, keepdims=True, axis=1) - np.min(new_all_img_token_path, keepdims=True, axis=1)))
    new_all_img_token_path = np.clip(new_all_img_token_path, a_min=0.01, a_max=0.99)
    new_all_img_token_path = {'img_'+str(n+1): [int(i // (1/num_expert) + 1) for i in j] for n, j in enumerate(new_all_img_token_path)}


    # new_all_text_token_path.update(new_all_img_token_path)
    # highlight dict --> team to highlight and their corresponding colors
    highlight_dict = {
        'text_1': "#BF5029",
        'text_2': "#FF9069"
    }

    # match-week
    match_day = [str(num*2+1) for num in range(num_moe_layers)]


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    # instantiate object
    bumpy = Bumpy(
        background_color="#FFFFFF", scatter_color="#808080",
        label_color="#000000", line_color="#C0C0C0",
        # rotate_xticks=90,  # rotate x-ticks by 90 degrees
        ticklabel_size=19, label_size=23,  # ticklable and label font-size
        scatter_points='o',   # other markers
        scatter_primary='D',  # marker to be used for teams
        scatter_size=150,   # size of the marker
        show_right=False,  # show position on the rightside
        plot_labels=True,  # plot the labels
        alignment_yvalue=0.5,  # y label alignment
        alignment_xvalue=0.5,   # x label alignment
    )

    # plot bumpy chart
    bumpy.plot(
        x_list=match_day,  # match-day or match-week
        y_list=list(range(1, num_expert+1))[::-1],  # position value from 1 to 20
        values=new_all_text_token_path,  # values having positions for each team
        secondary_alpha=0.2,   # alpha value for non-shaded lines/markers
        highlight_dict=highlight_dict,  # team to be highlighted with their colors
        # figsize=(16, 8),  # size of the figure
        # x_label='MoE layer idx',
        y_label='Expert idx',  # label name
        ylim=(0.8, num_expert+0.2),  # y-axis limit
        lw=2.5,   # linewidth of the connecting lines
        upside_down=True,
        ax=ax1,
        # fontproperties=font_normal.prop,   # fontproperties for ticklabels/labels
    )

    highlight_dict = {
        'img_1': "#365CBF",
        'img_2': "#76BEFF"
    }

    # plot bumpy chart
    bumpy.plot(
        x_list=match_day,  # match-day or match-week
        y_list=list(reversed(list(range(1, num_expert+1)))),  # position value from 1 to 20
        values=new_all_img_token_path,  # values having positions for each team
        secondary_alpha=0.2,   # alpha value for non-shaded lines/markers
        highlight_dict=highlight_dict,  # team to be highlighted with their colors
        # figsize=(16, 8),  # size of the figure
        x_label='MoE layer idx', y_label='Expert idx',  # label name
        ylim=(0.8, num_expert+0.2),  # y-axis limit
        lw=2.5,   # linewidth of the connecting lines
        upside_down=True,
        ax=ax2,
        # fontproperties=font_normal.prop,   # fontproperties for ticklabels/labels
    )


    legend_elements = [Line2D([0], [0], marker='D', ms=12, color='#BF5029', lw=3.5, label='Top-1'),
                       Line2D([0], [0], marker='D', ms=12, color='#FF9069', lw=3.5, label='Top-2'),
                       Line2D([0], [0], marker='o', ms=12, color='gray', lw=3.5, label='Others')]

    ax1.spines['top'].set_visible(True)
    ax1.spines['right'].set_visible(True)
    ax1.spines['left'].set_visible(True)
    ax1.spines['bottom'].set_visible(True)
    ax1.set_title('Text', fontsize=28)
    ax1.legend(handles=legend_elements, loc=(1.01, 0.25),
               ncol=1, fontsize=23)


    legend_elements = [Line2D([0], [0], marker='D', ms=12, color='#365CBF', lw=3.5, label='Top-1'),
                       Line2D([0], [0], marker='D', ms=12, color='#76BEFF', lw=3.5, label='Top-2'),
                       Line2D([0], [0], marker='o', ms=12, color='gray', lw=3.5, label='Others')]

    ax2.spines['top'].set_visible(True)
    ax2.spines['right'].set_visible(True)
    ax2.spines['left'].set_visible(True)
    ax2.spines['bottom'].set_visible(True)
    ax2.set_title('Image', fontsize=28)
    ax2.legend(handles=legend_elements, loc=(1.01, 0.25),
               ncol=1, fontsize=23)



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


import argparse
import shutil
import subprocess

import torch
import gradio as gr
from fastapi import FastAPI
import os
from PIL import Image
import tempfile
from decord import VideoReader, cpu
from transformers import TextStreamer

from moellava.conversation import conv_templates, SeparatorStyle, Conversation
from moellava.serve.gradio_utils import Chat, tos_markdown, learn_more_markdown, title_markdown, block_css

from moellava.constants import DEFAULT_IMAGE_TOKEN


def save_image_to_local(image):
    filename = os.path.join('temp', next(tempfile._get_candidate_names()) + '.jpg')
    image = Image.open(image)
    image.save(filename)
    # print(filename)
    return filename


def save_video_to_local(video_path):
    filename = os.path.join('temp', next(tempfile._get_candidate_names()) + '.mp4')
    shutil.copyfile(video_path, filename)
    return filename


def generate(image1, textbox_in, first_run, state, state_, images_tensor):

    print(image1)
    flag = 1
    if not textbox_in:
        if len(state_.messages) > 0:
            textbox_in = state_.messages[-1][1]
            state_.messages.pop(-1)
            flag = 0
        else:
            return "Please enter instruction"

    image1 = image1 if image1 else "none"
    # assert not (os.path.exists(image1) and os.path.exists(video))

    if type(state) is not Conversation:
        state = conv_templates[conv_mode].copy()
        state_ = conv_templates[conv_mode].copy()
        images_tensor = []

    first_run = False if len(state.messages) > 0 else True

    text_en_in = textbox_in.replace("picture", "image")

    image_processor = handler.image_processor
    if os.path.exists(image1):
        tensor = image_processor.preprocess(Image.open(image1).convert('RGB'), return_tensors='pt')['pixel_values'][0].to(handler.model.device, dtype=dtype)
        # print(tensor.shape)
        images_tensor.append(tensor)

    if os.path.exists(image1):
        text_en_in = DEFAULT_IMAGE_TOKEN + '\n' + text_en_in
    text_en_out, state_ = handler.generate(images_tensor, text_en_in, first_run=first_run, state=state_)
    state_.messages[-1] = (state_.roles[1], text_en_out)

    text_en_out = text_en_out.split('#')[0]
    textbox_out = text_en_out

    show_images = ""
    if os.path.exists(image1):
        filename = save_image_to_local(image1)
        show_images += f'<img src="./file={filename}" style="display: inline-block;width: 250px;max-height: 400px;">'
    if flag:
        state.append_message(state.roles[0], textbox_in + "\n" + show_images)
    state.append_message(state.roles[1], textbox_out)

    # return (state, state_, state.to_gradio_chatbot(), False, gr.update(value=None, interactive=True), images_tensor,
    #         gr.update(value=image1 if os.path.exists(image1) else None, interactive=True))
    return (state, state_, state.to_gradio_chatbot(), False, gr.update(value=None, interactive=True), images_tensor,
            gr.update(value=None, interactive=True))


def regenerate(state, state_):
    state.messages.pop(-1)
    state_.messages.pop(-1)
    if len(state.messages) > 0:
        return state, state_, state.to_gradio_chatbot(), False
    return (state, state_, state.to_gradio_chatbot(), True)


def clear_history(state, state_):
    state = conv_templates[conv_mode].copy()
    state_ = conv_templates[conv_mode].copy()
    return (gr.update(value=None, interactive=True),
            gr.update(value=None, interactive=True), \
            True, state, state_, state.to_gradio_chatbot(), [])

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default='LanguageBind/MoE-LLaVA-QWen-1.8B-4e2-1f')
parser.add_argument("--local_rank", type=int, default=-1)
args = parser.parse_args()

# import os
# required_env = ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
# os.environ['RANK'] = '0'
# os.environ['WORLD_SIZE'] = '1'
# os.environ['MASTER_ADDR'] = "192.168.1.201"
# os.environ['MASTER_PORT'] = '29501'
# os.environ['LOCAL_RANK'] = '0'
# if auto_mpi_discovery and not all(map(lambda v: v in os.environ, required_env)):

model_path = args.model_path

if 'qwen' in model_path.lower():  # FIXME: first
    conv_mode = "qwen"
elif 'openchat' in model_path.lower():  # FIXME: first
    conv_mode = "openchat"
elif 'phi' in model_path.lower():  # FIXME: first
    conv_mode = "phi"
elif 'stablelm' in model_path.lower():  # FIXME: first
    conv_mode = "stablelm"
else:
    conv_mode = "v1"
device = 'cuda'
load_8bit = False
load_4bit = False if 'moe' in model_path.lower() else True
dtype = torch.half
handler = Chat(model_path, conv_mode=conv_mode, load_8bit=load_8bit, load_4bit=load_4bit, device=device)
handler.model.to(dtype=dtype)
if not os.path.exists("temp"):
    os.makedirs("temp")

app = FastAPI()

textbox = gr.Textbox(
    show_label=False, placeholder="Enter text and press ENTER", container=False
)
with gr.Blocks(title='MoE-LLaVA🚀', theme=gr.themes.Default(), css=block_css) as demo:
    gr.Markdown(title_markdown)
    state = gr.State()
    state_ = gr.State()
    first_run = gr.State()
    images_tensor = gr.State()

    with gr.Row():
        with gr.Column(scale=3):
            image1 = gr.Image(label="Input Image", type="filepath")

            cur_dir = os.path.dirname(os.path.abspath(__file__))
            gr.Examples(
                examples=[
                    [
                        f"{cur_dir}/examples/extreme_ironing.jpg",
                        "What is unusual about this image?",
                    ],
                    [
                        f"{cur_dir}/examples/waterview.jpg",
                        "What are the things I should be cautious about when I visit here?",
                    ],
                    [
                        f"{cur_dir}/examples/desert.jpg",
                        "If there are factual errors in the questions, point it out; if not, proceed answering the question. What’s happening in the desert?",
                    ],
                ],
                inputs=[image1, textbox],
            )

        with gr.Column(scale=7):
            chatbot = gr.Chatbot(label="MoE-LLaVA", bubble_full_width=True).style(height=750)
            with gr.Row():
                with gr.Column(scale=8):
                    textbox.render()
                with gr.Column(scale=1, min_width=50):
                    submit_btn = gr.Button(
                        value="Send", variant="primary", interactive=True
                    )
            with gr.Row(elem_id="buttons") as button_row:
                upvote_btn = gr.Button(value="👍  Upvote", interactive=True)
                downvote_btn = gr.Button(value="👎  Downvote", interactive=True)
                flag_btn = gr.Button(value="⚠️  Flag", interactive=True)
                # stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=True)
                clear_btn = gr.Button(value="🗑️  Clear history", interactive=True)

    gr.Markdown(tos_markdown)
    gr.Markdown(learn_more_markdown)

    submit_btn.click(generate, [image1, textbox, first_run, state, state_, images_tensor],
                     [state, state_, chatbot, first_run, textbox, images_tensor, image1])

    regenerate_btn.click(regenerate, [state, state_], [state, state_, chatbot, first_run]).then(
        generate, [image1, textbox, first_run, state, state_, images_tensor],
        [state, state_, chatbot, first_run, textbox, images_tensor, image1])

    clear_btn.click(clear_history, [state, state_],
                    [image1, textbox, first_run, state, state_, chatbot, images_tensor])

# app = gr.mount_gradio_app(app, demo, path="/")
demo.launch()

# uvicorn llava.serve.gradio_web_server:app

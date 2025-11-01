import torch
from PIL import Image
from torchvision import transforms
import gradio as gr
from models import Encoder, Decoder

# =====================
# Device & Transform
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((667, 1000)),
    transforms.ToTensor()
])
to_pil = transforms.ToPILImage()

# =====================
# Load Models
# =====================
encoder = Encoder().to(device)
decoder = Decoder().to(device)

checkpoint = torch.load("checkpoints/model_epoch_10.pth", map_location=device)
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])
encoder.eval()
decoder.eval()

# =====================
# Functions
# =====================
def encrypt(cover_img, secret_img):
    cover = transform(cover_img).unsqueeze(0).to(device)
    secret = transform(secret_img).unsqueeze(0).to(device)

    with torch.no_grad():
        stego = encoder(cover, secret)

    return to_pil(stego.squeeze(0))

def decrypt(stego_img):
    stego = transform(stego_img).unsqueeze(0).to(device)

    with torch.no_grad():
        recovered = decoder(stego)

    return to_pil(recovered.squeeze(0))

# =====================
# Build Gradio Interface with Tabs
# =====================
with gr.Blocks(title="Image Steganography using Deep Learning") as demo:
    gr.Markdown("## Deep Learning Image Steganography")
    
    with gr.Tab("Encrypt"):
        cover_input = gr.Image(type="pil", label="Cover Image")
        secret_input = gr.Image(type="pil", label="Secret Image")
        encrypt_btn = gr.Button("Encrypt")
        stego_output = gr.Image(type="pil", label="Stego Image")
        encrypt_btn.click(fn=encrypt, inputs=[cover_input, secret_input], outputs=stego_output)
    
    with gr.Tab("Decrypt"):
        stego_input = gr.Image(type="pil", label="Stego Image")
        decrypt_btn = gr.Button("Decrypt")
        recovered_output = gr.Image(type="pil", label="Recovered Secret")
        decrypt_btn.click(fn=decrypt, inputs=stego_input, outputs=recovered_output)

if __name__ == "__main__":
    demo.launch()

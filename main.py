import telebot
import numpy as np
from PIL import Image
from model.config import DEVICE, GEN_CHECKPOINT_PATH, IMAGES_TO_PROCESS_PATH, transform_edge, LEARNING_RATE, BETA_1, BETA_2
from model.generator import Generator
from model.checkpoint_utils import load_checkpoint
import logging
import torch
from torchvision.utils import save_image

gen = Generator().to(DEVICE)
opt_gen = torch.optim.Adam(gen.parameters(), lr = LEARNING_RATE, betas=(BETA_1, BETA_2))
load_checkpoint(GEN_CHECKPOINT_PATH, gen, opt_gen)

logging.basicConfig(level=logging.DEBUG)

bot = telebot.TeleBot('6113236368:AAEqDpeCR0dvh433QcikeDJSDNwKg-D5apA')

@bot.message_handler(commands=["start"])
def start(m):
    bot.send_message(m.chat.id, 'Привет. Я дизайнер ботинок, кроссовок и всякой обуви, отправь мне скетч своей обувки, а я сделаю из этого реалистичную картинку!')


@bot.message_handler(content_types=['photo'])
def answer_to_photo(message):
    raw = message.photo[-1].file_id
    name = IMAGES_TO_PROCESS_PATH + raw + ".jpg"
    file_info = bot.get_file(raw)
    downloaded_file = bot.download_file(file_info.file_path)
    with open(name, 'wb') as new_file:
        new_file.write(downloaded_file)
    img_np = np.asarray(Image.open(name))
    img = transform_edge(image=img_np)['image']
    gen.eval()
    with torch.no_grad():
        y_fake = gen(img[None].to(DEVICE))
        y_fake = y_fake * 0.5 + 0.5
    res = np.rollaxis(y_fake[0].cpu().detach().numpy(), 0, 3)
    painted_img = Image.fromarray(np.uint8(res * 255))
    bot.send_photo(message.chat.id, painted_img)


bot.polling(none_stop=True, interval=0)
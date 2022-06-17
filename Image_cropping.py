from PIL import Image

for i in range(3):
    name = "animation.sat-%s.png"%i
    im = Image.open("D:/Users/rmalinow/Downloads/22-06-29 Presentation CNES CS Group/Images/GIF/" + name)
    im_bis = im.crop((0, 75, 718, 680))
    im_bis.save("D:/Users/rmalinow/Downloads/" + name)
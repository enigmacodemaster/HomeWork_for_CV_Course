from PIL import Image, ImageDraw, ImageFont

image = Image.open('Beauty.png')
text = '欧阳紫洲_CV学员'

font = ImageFont.truetype('C:\\Windows\\Fonts\\simhei.ttf',48)
# 图层转换为RGBA四通道
layer = image.convert('RGBA')

text_overlay = Image.new('RGBA', layer.size, (255,255,255,0))
image_draw = ImageDraw.Draw(text_overlay)

text_size_x, text_size_y = image_draw.textsize(text, font=font)
text_xy = (layer.size[0] - text_size_x*1.2, layer.size[1] - text_size_y*1.2)

image_draw.text(text_xy, text, font=font, fill=(255,120,255,90))

after = Image.alpha_composite(layer, text_overlay)

after.save('im_add_water.png')


'''
添加图片水印
'''

# # 图像
# img = Image.open('image1.jpg')
# # 水印图片
# logo = Image.open('logo.jpg')


# # 创建新的图层
# layer = Image.new('RGBA', img.size, (255,255,255,0))
# # 将logo图片粘贴到新建的图层上
# layer.paste(logo, (img.size[0] - logo.size[0], img.size[1] - logo[1]))
# # 覆盖
# img_after = Image.composite(layer, img, layer)
# img_after.show()
# img_after.save('target.jpg')

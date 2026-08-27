from PIL import Image

Image.MAX_IMAGE_PIXELS = None 

img_path = '/home/saurabhg/scratch/CORGIS_DATASET/flow_outputs/icfm/distances/iteration1/compare_images_sorted_distance.png'
img = Image.open(img_path)


new_size = (img.width // 4, img.height // 4)
resized_img = img.resize(new_size, Image.Resampling.LANCZOS)


resized_img.save('/home/saurabhg/scratch/CORGIS_DATASET/flow_outputs/icfm/distances/iteration1/compare_images_sorted_distance_viewable_grid.png', optimize=True, quality=85)
print("Saved viewable_grid.png!")
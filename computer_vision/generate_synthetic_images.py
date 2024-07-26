from PIL import Image 
import os 

save_dir = 'data/synthetic_images'
#create 
os.makedirs(save_dir , exist_ok= True)
#generate and save synthetic image
for i in range(100): #generate 100 images
    #create a new image for simplecity using solid color 
    random_color = tuple(np.random.randint([256 , 256 ,256]))
    image = Image.new('RGB'  , size = (256 ,256) , color = random_color)

    #save the image 
    image.save(os.path.join(save_dir , f'image_{i}.jpg'))

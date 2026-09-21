import torch
from tqdm import tqdm
import lpips
import torchvision
from torchvision.transforms import InterpolationMode
import os
from absl import flags,app
FLAGS=flags.FLAGS
flags.DEFINE_string('savedir','/home/saurabhg/scratch/FLOW_BASED_MODELS/LATENT_DATASET/flow_outputs_2/icfm/distances',help='folder where metrics will be saved')
flags.DEFINE_bool('calculate_lpips_modes',True,help='Do you want to run calculate lpips modes function')
flags.DEFINE_integer('iteration',1,help='iteration which need to be evaluated')
flags.DEFINE_string('original_images','/home/saurabhg/scratch/FLOW_BASED_MODELS/LATENT_DATASET/flow_outputs_2/icfm/image_tensor/original_image_tensor.pt',help='path to original image')
flags.DEFINE_string('generated_images','/home/saurabhg/scratch/FLOW_BASED_MODELS/LATENT_DATASET/flow_outputs_2/icfm/image_tensor/generated_images_1000_iteration1_python_inference_scipt.pt',help='filename of the generated_images_tensor')
def calculate_lpips_modes(generated_images, original_images):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Setting up LPIPS VGG...")
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    loss_fn_vgg.eval()
    original_images = original_images.to(device)
    indices = []
    total_images = len(generated_images)
    # batch_size = total_images // 10  
    batch_size=250
    print(f"Calculating distances in batches of {batch_size}")
    for i in tqdm(range(len(original_images))):
        orig_img = original_images[i:i+1] 
        best_dist = float('inf')
        best_idx = -1     
        for j in range(0, total_images, batch_size):
            gen_chunk = generated_images[j:j+batch_size].to(device)
            orig_expanded = orig_img.expand_as(gen_chunk)
            with torch.no_grad():
                distances = loss_fn_vgg(gen_chunk, orig_expanded).squeeze()
            chunk_closest_idx = torch.argmin(distances).item()
            chunk_best_dist = distances[chunk_closest_idx].item()
            if chunk_best_dist < best_dist:
                best_dist = chunk_best_dist
                best_idx = j + chunk_closest_idx 
            del gen_chunk, orig_expanded, distances  
        indices.append((best_idx, best_dist))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    indices = torch.tensor(indices)
    torch.save(indices,FLAGS.savedir+f'/iteration{FLAGS.iteration}/lpips_minimum_indices.pt')
    sorted_distances_indices = torch.sort(indices[:, 1])[1]
    sorted_distances_indices = torch.sort(indices[:, 1])[1]
    closest_generated_images = generated_images[indices[:, 0].long()].to(device)
    compare_image = torch.stack((original_images,closest_generated_images), dim=-1)
    compare_image = compare_image[sorted_distances_indices].permute([0, 4, 1, 2, 3]).reshape(-1, 3, 256, 256)
    
    torchvision.utils.save_image(
        compare_image, 
        FLAGS.savedir+f'/iteration{FLAGS.iteration}/compare_images_sorted_distance.png', 
        nrow=2
    )
    print('Done')


def calculate_lpips_modes_way2(generated_images,original_images,return_indices=False):
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    loss_fn_vgg=lpips.LPIPS(net='vgg').to(device)
    loss_fn_vgg.eval()

    print(f'Original Images:{original_images.shape}')
    print(f'Generated_images: {generated_images.shape}')
    
    batch_size=100 # Lowered to 100 for safety
    indices=[]
    
    for i in tqdm(range(len(generated_images))):
        best_index=0
        best_distance=float('inf')
        
        # Keep the single base image on CPU
        gen_img_base = generated_images[i:i+1]
        
        for j in range(0,len(original_images),batch_size):
            # Move only the small batch to the GPU
            original_image_batch = original_images[j:j+batch_size].to(device)
            generated_img = gen_img_base.expand_as(original_image_batch).to(device)
            
            with torch.no_grad():
                distances = loss_fn_vgg(generated_img, original_image_batch).to(device)
                
            closest_index = torch.argmin(distances).item()
            closest_distance = distances[closest_index].item()
        
            if closest_distance < best_distance:
                best_distance = closest_distance
                best_index = j + closest_index
                
            # Aggressive cleanup
            del distances, original_image_batch, generated_img
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        indices.append((best_index,best_distance))
        
    # Process final tensors on CPU
    indices=torch.tensor(indices)
    print(f'Indices:{indices.shape}')
    torch.save(indices,FLAGS.savedir+f'/iteration{FLAGS.iteration}/lpips_way_2_minimum_indices.pt')

    sorted_distance=torch.sort(indices[:,1]) 
    sorted_distances_indices=sorted_distance[1] 
    top_100=sorted_distances_indices#[:100]
    print(f'Sorted_distances_indices:{sorted_distances_indices.shape}')

    closest_original_images = original_images[indices[:,0].long()]
    print(f'Closest original Images:{closest_original_images.shape}')
    compare_image = torch.stack((generated_images, closest_original_images), dim=-1)
    print(f'Compare Image:{compare_image.shape}')
    compare_image = compare_image[top_100].permute([0,4,1,2,3]).reshape(-1,3,256,256)
    
    torchvision.utils.save_image(compare_image,FLAGS.savedir+f'/iteration{FLAGS.iteration}/compare_images_sorted_distance_python_way2.png',nrow=2)
    
    if return_indices:
        return indices
    print('Done')

def count_frequency(generated_images,original_images,return_frequency=True):
    indices=torch.load(FLAGS.savedir+f'/iteration{FLAGS.iteration}/lpips_way_2_minimum_indices.pt')

    indices=indices.to(torch.device('cpu'))
    out=torch.unique(indices[:,0],return_counts=True)
    out=torch.stack((out[0],out[1]),dim=-1)
    dummy_tensor=torch.stack((torch.arange(0,len(original_images),1),torch.zeros(len(original_images))),dim=1)
    row_indices,values=out[:,0],out[:,1]
    dummy_tensor[row_indices.long(),1]=values
    
    frequency_sorted_indexes=torch.argsort(dummy_tensor[:,1])
    out=dummy_tensor[frequency_sorted_indexes]
    generated_images_sorted_by_frequency=original_images[out[:,0].long()]
    torchvision.utils.save_image(generated_images_sorted_by_frequency,FLAGS.savedir+f'/iteration{FLAGS.iteration}/{len(original_images)}_generated_images_sorted_through_frequency_script.png')

    torch.save(dummy_tensor,FLAGS.savedir+f'/iteration{FLAGS.iteration}/frequency_distribution.pt')

    if return_frequency:
        return dummy_tensor



def frequency_mask(minimum_indices_way1,threshold):
    minimum_indices_way1=minimum_indices_way1 if isinstance(minimum_indices_way1,torch.Tensor) else torch.load('minimum_indices_way_1')
    distance_values=minimum_indices_way1[:,1]
    mask=distance_values>threshold
    mask=(~mask.bool()).int()
    return mask

def evaluate(argv):
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if device==torch.device('cuda'):
        original_images=torch.load(FLAGS.original_images)
        print(FLAGS.generated_images)
        generated_images=torch.load(FLAGS.generated_images)
    original_images=torch.load(FLAGS.original_images,map_location=torch.device('cpu'))
    print(FLAGS.generated_images)
    generated_images=torch.load(FLAGS.generated_images,map_location=torch.device('cpu'))

    if FLAGS.calculate_lpips_modes:

        calculate_lpips_modes(generated_images,original_images)
    # calculate_lpips_modes_way2(generated_images,original_images)
    count_frequency(generated_images,original_images)



if __name__=="__main__":
    app.run(evaluate)

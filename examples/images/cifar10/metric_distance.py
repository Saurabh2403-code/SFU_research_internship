import torch
from tqdm import tqdm
import lpips
import torchvision
from torchvision.transforms import InterpolationMode
import os
# from absl import flags,apps
# def calculate_lpips_modes(generated_images,original_images):
#     device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     loss_fn_vgg=lpips.LPIPS(net='vgg').to(device)
#     loss_fn_vgg.eval()
#     original_images=original_images.to(device)
#     indices=[]
#     total_images=len(generated_images)
#     batch_size=total_images//10
#     for i in tqdm(range(len(original_images))):
#         generated_images=generated_images.to(device)
#         for j in range(0,total_images,batch_size):
#             generated_images_chunked=generated_images_chunked[j:j+batch_size]
#             original_img=original_images[i].expand_as(generated_images_chunked)
#             with torch.no_grad():
#                 distances=loss_fn_vgg(generated_images,original_img).to(device)
#         closest_idx=torch.argmin(distances).item()
#         indices.append((closest_idx,distances[closest_idx].item()))
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#     indices=torch.tensor(indices)
#     torch.save(indices,'/home/saurabhg/scratch/LATENT_DATASET/flow_outputs/icfm/distances/iteration2/lpips_minimum_indices.pt')

    # # sorted_distance=torch.sort(indices[:,1])
    # # sorted_distances_indices=sorted_distance[1]
    # sorted_distances_indices=torch.sort(indices[:,1])[1]
    # closest_generated_images=generated_images[indices[:,0].long()]
    # compare_image=torch.stack((original_images,closest_generated_images),dim=-1)
    # compare_image=compare_image[sorted_distances_indices].permute([0,4,1,2,3]).reshape(-1,3,256,256)
    # torchvision.utils.save_image(compare_image,'/home/saurabhg/scratch/LATENT_DATASET/flow_outputs/icfm/distances/iteration2/compare_89_images_sorted_distance_python_2.png',nrow=2)
    # print('Done')
    # # unique_modes=len(set(indices))
    # # return unique_modes
def calculate_lpips_modes(generated_images, original_images):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    print("Setting up LPIPS VGG...")
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    loss_fn_vgg.eval()
    
 
    # if original_images.min() >= 0.0:
    #     original_images = original_images * 2.0 - 1.0
    # if generated_images.min() >= 0.0:
    #     generated_images = generated_images * 2.0 - 1.0
        
    original_images = original_images.to(device)
    indices = []
    
    total_images = len(generated_images)
    batch_size = total_images // 10  
    
    print("Calculating distances using chunked inner loop...")
  
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
    

    os.makedirs('/home/saurabhg/scratch/LATENT_DATASET/flow_outputs/icfm/distances/iteration3/', exist_ok=True)
    torch.save(indices, '/home/saurabhg/scratch/LATENT_DATASET/flow_outputs/icfm/distances/iteration3/lpips_minimum_indices.pt')

    sorted_distances_indices = torch.sort(indices[:, 1])[1]
    

    closest_generated_images = generated_images[indices[:, 0].long()].to(device)
    

    # orig_vis = original_images * 0.5 + 0.5
    # gen_vis = closest_generated_images * 0.5 + 0.5
    
    compare_image = torch.stack((original_images,closest_generated_images), dim=-1)
    
   
    compare_image = compare_image[sorted_distances_indices].permute([0, 4, 1, 2, 3]).reshape(-1, 3, 256, 256)
    
    torchvision.utils.save_image(
        compare_image, 
        '/home/saurabhg/scratch/LATENT_DATASET/flow_outputs/icfm/distances/iteration3/compare_images_sorted_distance.png', 
        nrow=2
    )
    print('Done')

def calculate_lpips_modes_way2(generated_images,original_images,return_indices=False):
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    loss_fn_vgg=lpips.LPIPS(net='vgg').to(device)
    loss_fn_vgg.eval()
    generated_images=generated_images.to(device)
    original_images=original_images.to(device)
    print(f'Original Images:{original_images.shape}')
    print(f'Generated_images: {generated_images.shape}')
    generated_images=generated_images.to(device)
    indices=[]
    for i in tqdm(range(len(generated_images))):
        generated_img=generated_images[i].expand_as(original_images)
        with torch.no_grad():
            distances=loss_fn_vgg(generated_img,original_images).to(device)
        closest_idx=torch.argmin(distances).item()
        indices.append((closest_idx,distances[closest_idx].item()))
    indices=torch.tensor(indices).to(device)
    print(f'Indices:{indices.shape}')
    torch.save(indices,'/home/saurabhg/scratch/LATENT_DATASET/flow_outputs/icfm/distances/iteration1/lpips_way_2_minimum_indices.pt')

    sorted_distance=torch.sort(indices[:,1]) 
    sorted_distances_indices=sorted_distance[1] #shape=[1000,1]
    print(f'Sorted_distances_indices:{sorted_distances_indices.shape}')

    # sorted_distances_indices=torch.sort(indices[:,1])[1]
    # original_images=torch.tile(original_images,(10,1,1,1))
    closest_original_images=original_images[indices[:,0].long()]
    print(f'Closest original Images:{closest_original_images.shape}')
    compare_image=torch.stack((generated_images,closest_original_images),dim=-1)
    print(f'Compare Image:{compare_image.shape}')
    compare_image=compare_image[sorted_distances_indices].permute([0,4,1,2,3]).reshape(-1,3,256,256)
    torchvision.utils.save_image(compare_image,'/home/saurabhg/scratch/LATENT_DATASET/flow_outputs/icfm/distances/iteration1/compare_images_sorted_distance_python_way2.png',nrow=2)
    if return_indices:
        return indices
    print('Done')

# class Cosine_Similarity:
#     def __init__(self,image1_path,image2_path):
#         device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#         self.image1_path=image1_path
#         self.image2_path=image2_path
#     def model(self):
#         weights=torchvision.models.ViT_H_14_Weights.DEFAULT
#         self.model=torchvision.models.vit_h_14(weights=weights)
#     def process_image(self,image_path):
#         image=torch.load(self.image1_path)
#         transform=torchvision.transforms.Compose(
#             [
#                 torchvision.transforms.Resize((518, 518), interpolation=InterpolationMode.BICUBIC),
#                 torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

#             ]
#         )
#         return transform(image)
#     def get_emb(self):

#         img1,img2=self.process_image(self.image1_path),self.process_image(self.image2_path)
#         emb1,emb2=self.model(img1),self.model(img2)
#         return emb1.detach().cpu(),emb2.detach().cpu()
#     def score(self):
#         emb1,emb2=self.get_emb()
#         scores=torch.nn.funnctional.cosine_similarity(emb1,emb2)
#         return scores.numpy().tolist()



def count_frequency(generated_images,original_images,return_frequency=False):
    # indices=calculate_lpips_modes_way2(generated_images,original_images)
    indices=torch.load('/home/saurabhg/scratch/LATENT_DATASET/flow_outputs/icfm/distances/iteration1/lpips_way_2_minimum_indices.pt')
    indices=indices.to(torch.device('cpu'))
    out=torch.unique(indices[:,0],return_counts=True)
    out=torch.stack((out[0],out[1]),dim=-1)
    indexes=torch.argsort(out[:,-1])
    out=out[indexes]
    generated_images_sorted_by_frequency=original_images[out[:,0].long()]
    torchvision.utils.save_image(generated_images_sorted_by_frequency,'/home/saurabhg/scratch/LATENT_DATASET/flow_outputs/icfm/distances/iteration1/10000_generated_images_sorted_through_frequency_script.png')
    torch.save(out,'/home/saurabhg/scratch/LATENT_DATASET/flow_outputs/icfm/distances/iteration1/frequency_distribution.pt')
    if return_frequency:
        return out

 


original_images=torch.load('/home/saurabhg/scratch/LATENT_DATASET/flow_outputs/icfm/image_tensor/original_image_tensor.pt')
generated_images=torch.load('/home/saurabhg/scratch/LATENT_DATASET/flow_outputs/icfm/image_tensor/generated_images_1000_iteration3_python_inference_scipt.pt')
calculate_lpips_modes(generated_images,original_images)
# calculate_lpips_modes_way2(generated_images,original_images)
# count_frequency(generated_images,original_images)

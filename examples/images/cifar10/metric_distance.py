import torch
from tqdm import tqdm
import lpips
import torchvision
def calculate_lpips_modes(generated_images,original_images):
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    loss_fn_vgg=lpips.LPIPS(net='vgg').to(device)
    loss_fn_vgg.eval()
    original_images=original_images.to(device)
    indices=[]
    for i in tqdm(range(len(original_images))):
        generated_images=generated_images.to(device)
        original_img=original_images[i].expand_as(generated_images)
        with torch.no_grad():
            distances=loss_fn_vgg(generated_images,original_img).to(device)
        closest_idx=torch.argmin(distances).item()
        indices.append((closest_idx,distances[closest_idx].item()))
    indices=torch.tensor(indices)
    torch.save(indices,'/home/saurabhg/scratch/flow_outputs/icfm/distances/lpips_minimum_indices.pt')

    sorted_distance=torch.sort(indices[:,1])
    sorted_distances_indices=sorted_distance[1]

    sorted_distances_indices=torch.sort(indices[:,1])[1]
    closest_generated_images=generated_images[indices[:,0].long()]
    compare_image=torch.stack((original_images,closest_generated_images),dim=-1)
    compare_image=compare_image[sorted_distances_indices].permute([0,4,1,2,3]).reshape(-1,3,32,32)
    torchvision.utils.save_image(compare_image,'/home/saurabhg/scratch/flow_outputs/icfm/distances/compare_images_sorted_distance_python.png',nrow=2)
    print('Done')
    # unique_modes=len(set(indices))
    # return unique_modes

generated_samples = torch.load('/home/saurabhg/scratch/flow_outputs/icfm/image_tensor/generated_ICFM_images_step_90000.pt')
original_dataset = torch.load('/home/saurabhg/scratch/cifar10_data/image_tensor/original_image_100.pt')
calculate_lpips_modes(generated_samples,original_dataset)


import os
import json
import csv
import sys
from tqdm import tqdm
import torch
from PIL import Image
from torchvision import transforms
from model import swin_base_patch4_window12_384_in22k as create_model

def process_images_dual_weights(folder_path, output_csv):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = 384
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # read class_indict
    json_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'class_indices.json')
    assert os.path.exists(json_path), f"file: '{json_path}' does not exist."

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # Define both model weight paths
    weight_path1 = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'weights', 'modelT2-aaaa.pth') 
    weight_path2 = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'weights', 'modelT3-F.pth')

    # First Run
    model = create_model(num_classes=2).to(device)
    model.load_state_dict(torch.load(weight_path1, map_location=device))
    model.eval()

    first_run_results = {}

    # Process images with first weight
    img_names = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])
    
    # First run and save to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        for img_name in tqdm(img_names, desc="Processing with first weight"):
            img_path = os.path.join(folder_path, img_name)
            
            try:
                img = Image.open(img_path).convert('RGB')
                img = data_transform(img)
                img = torch.unsqueeze(img, dim=0)

                with torch.no_grad():
                    output = torch.squeeze(model(img.to(device))).cpu()
                    predict = torch.softmax(output, dim=0)
                    predict_cla = torch.argmax(predict).numpy()

                predicted_class = "1" if predict_cla != 0 else "0"
                first_run_results[os.path.splitext(img_name)[0]] = predicted_class
                csvwriter.writerow([os.path.splitext(img_name)[0], predicted_class])

            except Exception as e:
                print(f"Error processing {img_name}: {str(e)}")

    print("第一段结束，第二段开启")

    # Second Run
    model.load_state_dict(torch.load(weight_path2, map_location=device))
    model.eval()

    # Process images with second weight and compare
    updated_results = []
    
    for img_name in tqdm(img_names, desc="Processing with second weight"):
        img_path = os.path.join(folder_path, img_name)
        base_name = os.path.splitext(img_name)[0]
        
        try:
            img = Image.open(img_path).convert('RGB')
            img = data_transform(img)
            img = torch.unsqueeze(img, dim=0)

            with torch.no_grad():
                output = torch.squeeze(model(img.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()

            second_predicted_class = "1" if predict_cla != 0 else "0"
            
            # Compare with first run results
            if first_run_results[base_name] == "0" and second_predicted_class == "1":
                updated_results.append([base_name, "1"])
            else:
                updated_results.append([base_name, first_run_results[base_name]])

        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")
            updated_results.append([base_name, first_run_results.get(base_name, "0")])

    # Write final results to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(updated_results)

    print(f"Final results saved to {output_csv}")

if __name__ == '__main__':  
        # 检查操作系统类型
    if sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
        # Unix-like 系统（Linux, macOS）
        input_folder = '/testdata'
    else:
        # Windows 系统
        input_folder = r"D:\BaiduNetdiskDownload\C2\C2"

    # 确保输入目录存在
    assert os.path.exists(input_folder), f"Input folder does not exist: {input_folder}"

    # 输出CSV文件名
    output_csv = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cla_pre.csv')

    process_images_dual_weights(input_folder, output_csv)
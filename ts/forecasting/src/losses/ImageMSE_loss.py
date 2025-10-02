import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from scipy.spatial.distance import cosine
from sklearn.preprocessing import MinMaxScaler

class ImageMSE(nn.Module):
    def __init__(self):
        super(ImageMSE, self).__init__()
        self.mse = torch.nn.MSELoss()
        self.mse_image = torch.nn.MSELoss()
        
    def _plot_to_image(self, y_min, y_max, input):
        # # Batch * Dim
        buffer = BytesIO()
        
        plt.figure(facecolor='black')
        plt.gca().axes.xaxis.set_visible(False)
        plt.gca().axes.yaxis.set_visible(False)
        plt.plot(input, linewidth=1, color='black')  # 흰색 선으로 그래프 그리기

        plt.ylim(y_min, y_max)  # y축 범위 설정
        plt.tight_layout()

        # 이미지 저장
        plt.savefig(buffer, format='jpeg', facecolor=plt.gcf().get_facecolor())  # 배경 색상 유지
        plt.close()
        
        buffer.seek(0)
        image = Image.open(buffer).convert('L')
        return image
    
    def _cal_image_loss(self, pred, target):
        loss_list = []
        # Batch * Dim
        for batch in range(pred.shape[0]):
            for feature in range(pred.shape[-1]):
                input_target = target[batch,:,feature].detach().cpu()
                input_pred = pred[batch,:,feature].detach().cpu()
                y_min, y_max = input_target.min(), input_target.max()
                image_target = self._plot_to_image(y_min, y_max, input_target)
                image_pred = self._plot_to_image(y_min, y_max, input_pred)
                vec_target = np.array(image_target).astype(np.float32)
                vec_pred = np.array(image_pred).astype(np.float32)
                
                scaler = MinMaxScaler(feature_range=(-1, 1))
                scaler.fit(vec_target)
                vec_target = scaler.transform(vec_target)
                vec_pred = scaler.transform(vec_pred)
                
                # vec_target = torch.from_numpy(vec_target).to(pred.device)
                # vec_pred = torch.from_numpy(vec_pred).to(pred.device)

                vec_target = torch.tensor(vec_target, device=pred.device, dtype=torch.float32, requires_grad=True)
                vec_pred = torch.tensor(vec_pred, device=pred.device, dtype=torch.float32, requires_grad=True)
                
                imamge_similarity = self.mse_image(vec_target, vec_pred)
                loss_list.append(imamge_similarity)
        return loss_list

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        image_loss_list = self._cal_image_loss(pred, target)
        image_loss = torch.mean(torch.stack(image_loss_list))
        print(f'MSE Loss:{mse_loss}')
        print(f'Image Loss:{image_loss}')
        return mse_loss + image_loss
            
            

# class ImageMSE(nn.Module):
#     def __init__(self):
#         super(ImageMSE, self).__init__()
#         self.mse = torch.nn.MSELoss()
#         self.cos = torch.nn.CosineEmbeddingLoss()
        
#     def _plot_to_image(self, y_min, y_max, input):
#         # # Batch
#         buffer = BytesIO()
        
#         plt.gca().axes.xaxis.set_visible(False)
#         plt.gca().axes.yaxis.set_visible(False)
#         plt.imshow(input.detach().cpu())
#         plt.savefig(buffer, format='jpeg')
#         plt.close()
        
#         buffer.seek(0)
#         image = Image.open(buffer)
        
#         return image

#     def _image_to_vector(self, image):
#         return np.array(image).flatten()
    
#     def _cal_image_loss(self, pred, target):
#         loss_list = []
#         # Batch
#         for batch in range(pred.shape[0]):
#             input_target = target[batch,:].detach().cpu()
#             input_pred = pred[batch,:].detach().cpu()
#             y_min, y_max = input_target.min(), input_target.max()
#             image_target = self._plot_to_image(y_min, y_max, input_target)
#             image_pred = self._plot_to_image(y_min, y_max, input_pred)
#             vec_target = self._image_to_vector(image_target)
#             vec_pred = self._image_to_vector(image_pred)
#             breakpoint()
#             similarity = self.cos(vec_target, vec_pred)
#             loss_list.append(similarity)
#         return loss_list

#     def forward(self, pred, target):
#         mse_loss = self.mse(pred, target)
#         image_loss_list = self._cal_image_loss(pred, target)
#         print(f'MSE Loss:{mse_loss}')
#         print(f'Image Loss:{np.mean(image_loss_list)}')
#         return mse_loss + torch.tensor(np.mean(image_loss_list), device=pred.device, requires_grad=True)
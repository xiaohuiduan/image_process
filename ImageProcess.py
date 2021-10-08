import imageio
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import shape
from numpy.lib import imag, math, pad
import time


class ImageProcess:

    def covert_img_to_array(self, path:str) -> np.array:
        """[将图片转成Array便于处理]

        Args:
            path (str): [图片保存位置]

        Returns:
            np.array: [返回numpy数组，数组元素uint8]
        """
        return np.array(imageio.imread(path))

    def flip_180(self, arr: np.array) -> np.array:
        return arr[::-1,::-1]

    def show_img(self,title:str, imgs:list, cmaps:list,row:int = 0,col:int = 0):
        """展示图片 len(imgs) must equal to the len of cmaps

        Args:
            title (str): [图像标题]
            imgs (list): [图片元组]
            cmaps (list): [mask,plt以何种形式展示图片，可参考官方文档使用：'gray'表示灰度图，None表示彩色图]
            row (int, optional): [指令row]. Defaults to 0.
            col (int, optional): [指令col]. Defaults to 0.
        """
        if len(imgs) != len(cmaps):
            print("图片和mask的len必须相同")
        else:
            if row == 0 and col !=0:
                row = np.ceil(len(imgs)/col).astype("uint8")
            elif row!=0 and col == 0:
                col = np.ceil(len(imgs)/row).astype("uint8")
            elif row*col < len(imgs):
                # 尽量以方正的形式去展示图片
                row = np.ceil(np.sqrt(len(imgs))).astype("uint8")
                col = np.ceil(len(imgs)/row).astype("uint8")

            for index,img in enumerate(imgs):
                plt.subplot(row,col,index+1)
                plt.imshow(img,cmap=cmaps[index])
            plt.suptitle(title)
            plt.show()

    def scan_gray_img_line(self, img:np.array, I:int, loc:str) -> list:
        """黑白图像灰度扫描

        Args:
            img (np.array): [灰度图片]
            I (int): [行or列数]
            loc (str): [位置，行or列]

        Returns:
            Array: [对应行or列的灰度值，如果指定错误默认返回column]
        """
        I = int(I)
        if loc == "row":
            return img[I].reshape(1,img.shape[1])
        else:
            return img[:,I].reshape(img.shape[0],1)

    def covert_rgb_to_gray(self, image:np.array, method:str = 'NTSC') -> np.array:
        """将RGB图像转成gray图像

        Args:
            image (np.array): [rgb图像]
            method (str, optional): [转换模式]. Defaults to 'NTSC'.

        Returns:
            Array: [返回的灰度图像]
        """
        if method == 'average':
            gray_img = image[:,:,0]/3+image[:,:,1]/3+image[:,:,2]/3
        
        else:
            gray_img = image[:,:,0]*0.2989 + image[:,:,1]*0.5870 + image[:,:,2]*0.1140
        return gray_img
    
    def __matrix_dot_product(self,matrix,kernel):
        """矩阵点乘 [1,2,3]*[4,5,6] = 1*4 + 2*5 + 3*6 = 32

        Args:
            matrix ([type]): [部分图像]
            kernel ([type]): [kernel]

        Returns:
            [type]: [点乘结果]
        """
        if len(matrix) != len(kernel):
            print("点积失败，大小不一致")
        else:
            return (np.multiply(matrix,kernel)).sum()

            # result = 0
            # for i, row_nums in enumerate(matrix):
            #     for j,num in enumerate(row_nums):
            #         result += num * kernel[i][j]
            # return result 


    def __padding(self, padding_type:str, image:np.array, padding_w:int, padding_h:int):
        """对图片进行padding

        Args:
            padding_type (str): [padding方式]
            image (np.array): [图片]
            padding_w (int): [宽度pdding]
            padding_h (int): [高度padding，一般来说padding_w = padding_h]

        Returns:
            [type]: [返回padding之后的结果]
        """
        image_w = image.shape[0]
        image_h = image.shape[1]

        padding_image = np.zeros((image_w+padding_w*2,image_h+padding_h*2))
        padding_image[padding_w:padding_w+image_w,padding_h:padding_h+image_h] = image

        if padding_type == 'zero':
            return padding_image

        if padding_type == "replicate": 
            # 补充四个角
            padding_image[0:padding_w+1,0:padding_h+1] = image[0,0]
            padding_image[image_w+padding_w-1:,0:padding_h+1] = image[image_w-1,0]
            padding_image[0:padding_w+1,image_h+padding_h-1:] = image[0,image_h-1]
            padding_image[image_w+padding_w-1:,image_h+padding_h-1:] = image[image_w-1,image_h-1]

            # 补充旁边的元素
            for i in range(padding_w+1,image_w+padding_w-1):
                padding_image[i,0:padding_h] = image[i-padding_w,0]
                padding_image[i,image_h+padding_h:] = image[i-padding_w,image_h-1]
                
            
            for i in range(padding_h+1,image_h+padding_h-1):
                padding_image[0:padding_w,i] = image[0,i-padding_h]
                padding_image[image_w+padding_w:,i] = image[image_w-1,i-padding_h]
            return padding_image

    
    def corr2D(self, image:np.array, kernel:np.array, padding:str = 'zero') -> np.array:
        """对图片进行相关运算。

        Args:
            image (np.array): [(*,*)shape的图片]
            kernel (np.array): [kernel，kernel为奇数]
            padding (str, optional): [zero以零填充，replicate以邻近的填充]. Defaults to 'zero'.

        Returns:
            [type]: [description]
        """
        kernel_size_w = kernel.shape[0]
        kernel_size_h = kernel.shape[1]

        image_w,image_h = image.shape
        
        
        padding_w = kernel_size_w // 2
        padding_h = kernel_size_h // 2

        # 将图片padding起来
        padding_image = self.__padding(padding,image,padding_w,padding_h)

        new_image = np.zeros((image_w,image_h))
        for i in range(image_w):
            for j in range(image_h):
                new_image[i][j] = self.__matrix_dot_product(padding_image[i:i+kernel_size_w,j:j+kernel_size_h],kernel)

        return new_image.clip(0,255).astype("uint8")


    def conv2D(self, image:np.array, kernel:np.array, padding:str = 'zero') -> np.array:
        """二维卷积

        Args:
            image (np.array): [(*,*)shape的图片]
            kernel (np.array): [kernel，kernel为奇数]
            padding (str, optional): [zero以零填充，replicate以邻近的填充]. Defaults to 'zero'.

        Returns:
            [type]: [卷积好的结果]
        """
        return self.corr2D(image,self.flip_180(kernel),padding)
        

    def cov2D_color(self, image:np.array, kernel:np.array, padding:str = 'zero') -> np.array:
        """三维卷积

        Args:
            image (np.array): [(*,*,3)维度的图片]
            kernel (np.array): [kernel，kernel为奇数]
            padding (str, optional): [zero以零填充，replicate以邻近的填充]. Defaults to 'zero'.

        Returns:
            [type]: [卷积好的图片]
        """
        R = image[:,:,0]
        G = image[:,:,1]
        B = image[:,:,2]

        R_cov = self.conv2D(R,kernel,padding=padding)
        G_cov = self.conv2D(G,kernel,padding=padding)
        B_cov = self.conv2D(B,kernel,padding=padding)

        return np.dstack([R_cov,G_cov,B_cov])
    
    def gauss_2d_kernel(self,sig,m=0):
        """产生高斯核

        Args:
            sig ([type]): [高斯核参数]
            m (int, optional): [高斯kernel的大小]. Defaults to 0. if m=0，then m = ceil(3*sig)*2 +1

        Returns:
            [type]: [m*m大小的高斯核]
        """
        fit_m = math.ceil(3 * sig)*2+1

        if m == 0:
            m = fit_m
        if m < fit_m:
            print("你的核的size应该大一点")
        
        # 中心点
        center = m //2
        kernel = np.zeros(shape=(m,m))
        for i in range(m):
            for j in range(m):
                kernel[i][j] = (1/(2*math.pi*sig**2))*math.e**(-((i-center)**2+(j-center)**2)/(2*sig**2))

        return kernel/(kernel.sum())

        
        
        




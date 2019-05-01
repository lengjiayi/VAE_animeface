一个学习用的Deep Auto-Encoder模型，使用了[这篇文章](https://new.qq.com/omn/20180414/20180414A0GJMS.html)提供的动漫人脸数据集，为了加速训练我将图片转为了30\*30的灰度图片。

- 共有四层编码和解码器
- 训练时采用了MSE误差，Adam优化器
- optg.pth和vaeg.pth提供了训练好的优化器和VAE网络
- RandomGan中有随机产生的100张动漫人脸

### 效果

相比于GAN，VAE得生成结果会很模糊，并且可能会出现比较明显的噪点。我在实际训练时发现Conv生成得图片更模糊，可能是因为没有加入反向池化。

#### 编码-解码

![VAEdecode](https://github.com/lengjiayi/VAE_animeface/tree/master/assets/VAEdecode.PNG)

上图是全连接的VAE对18张随机的人脸编码-解码后的结果，可以看出对颜色的还原比较精准，但是对人脸倾斜角度的还原不是很好。

#### 生成

![VAEfixin](https://github.com/lengjiayi/VAE_animeface/tree/master/assets/VAEfixin.PNG)

上图是随机选取两个样本编码及其高维线段中间的七个等分点产生的脸，两端是输入图片，可以看到人脸的发色逐渐变浅，人脸的朝向也从向右逐渐转为向左。

![VAEgen](https://github.com/lengjiayi/VAE_animeface/tree/master/assets/VAEgen.PNG)

上图则是在编码空间使用numpy.random.normal随机产生的一些人脸，可以看到72.png出现了之前所说的明显噪点。


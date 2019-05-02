一个学习用的Deep Auto-Encoder模型，使用了[这篇文章](https://new.qq.com/omn/20180414/20180414A0GJMS.html)提供的动漫人脸数据集，为了加速训练我将图片转为了30\*30的灰度图片。

- 共有四层编码和解码器
- 训练时采用了MSE误差，Adam优化器
- optg.pth和vaeg.pth提供了训练好的优化器和VAE网络
- RandomGan中有随机产生的100张动漫人脸

### 效果

相比于GAN，VAE得生成结果会很模糊，并且可能会出现比较明显的噪点。我在实际训练时发现Conv生成得图片更模糊，可能是因为没有加入反向池化。

#### 编码-解码

![VAEdecode](https://github.com/lengjiayi/VAE_animeface/blob/master/assets/VAEdecode.PNG)

上图是全连接的VAE对18张随机的人脸编码-解码后的结果，可以看出对颜色的还原比较精准，但是对人脸倾斜角度的还原不是很好。

#### 生成

![VAEfixin](https://github.com/lengjiayi/VAE_animeface/blob/master/assets/VAEfixin.PNG)

上图是随机选取两个样本编码及其高维线段中间的七个等分点产生的脸，两端是输入图片，可以看到人脸的发色逐渐变浅，人脸的朝向也从向右逐渐转为向左。

![VAEgen](https://github.com/lengjiayi/VAE_animeface/blob/master/assets/VAEgen.PNG)

上图则是在编码空间使用numpy.random.normal随机产生的一些人脸，可以看到72.png出现了之前所说的明显噪点。

#### 编码空间

我试图解读每一个维度的作用，所以按照编码空间大小产生了十个序列的图片，分别对应十个维度的变化，如下图：

![VAEchannels](https://github.com/lengjiayi/VAE_animeface/blob/master/assets/VAEchannels.PNG)

事实证明这种非条件的学习不能韩浩的解读大部分维度。

- 第一、二、六、八、十维看起来就对生成结果没有什么影响。
- 第三维度增加时头发变短
- 第四维增加时阴影从左边跑到右边，不过不知道这是什么
- 第九维增加时头发颜色变浅

- 大部分维度都对脸的朝向有影响
## Pytorch学习过程记录
* tensor.f()一般是产生一个新的tensor,新tensor可能和原tensor共用storage
* tensor.f_()则是直接对原tensor进行操作
* tensor.gather(dim, index_tensor):
  返回一个和index_tensor一样形状的张量ret_tensor。
  ret_tensor的各元素取自tensor,index_tensor的对应元素作为指定dim上的索引值。
  比如dim=1是,若index_tensor为二维张量,则index_tensor[i][j]=tensor[i][index_tensor[i][j]]
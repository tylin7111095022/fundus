###################################################################################################################################
# gradCAM implement referred to the article : https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
# @Auther: TSUNG-YU LIN
# Edited : 2023-03-22

# gradCAM 本質上是將每個feature map上的梯度做global mean pooling來當作該張feature map的權重，並將所有feature map的值相加起來取正值並正規化
# 按照SPnet模型架構我將取 Risudual_GC_Module所輸出的特徵圖來進行gradCAM

###################################################################################################################################

import torch.nn as nn
import torch
import torch.nn.functional as F

class Swish(nn.Module):
    def __init__(self,inplace=True):
        super(Swish,self).__init__()
        self.inplace = inplace
    
    def forward(self,x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x*torch.sigmoid(x)
    
class FCN(nn.Module):
    def __init__(self,in_neuron:int, out_neuron:int,num_hidden_layer:int, reduction_rate:int = 4):
        super(FCN,self).__init__()
        hidden = [(nn.Linear(in_neuron//(reduction_rate**i), in_neuron//(reduction_rate**(i+1))),nn.ReLU()) for i in range(1,num_hidden_layer)]
        h = []
        for pair in hidden:
            l,a = pair
            h.append(l)
            h.append(a)

        self.input_layer = nn.Linear(in_neuron, in_neuron//reduction_rate)
        if h == []:
            self.hidden_layer = None
        else:
            self.hidden_layer = nn.Sequential(*h)
        self.output_layer = nn.Linear(in_neuron//(reduction_rate**num_hidden_layer), out_neuron)
    
    def forward(self,x):
        out = torch.relu(self.input_layer(x))
        if (self.hidden_layer is not None):
            out = self.hidden_layer(out)
        logits = self.output_layer(out)
        return logits
    

############### --Devider-- ################    

class Risudual_GC_Module(nn.Module):
    def __init__(self,in_ch:int,filters:int,img_shape:tuple):
        '''
        filters 按照原始paper 預設為32
        img_shape (h,w)
        '''
        super(Risudual_GC_Module,self).__init__()
        self.conv1 = nn.Conv2d(in_ch, filters,kernel_size=3,stride=1,padding=1)
        self.batchnorm1 = nn.BatchNorm2d(filters)
        self.swish1 = Swish()
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3,stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(filters)
        self.conv_1x1_1 = nn.Conv2d(in_ch, filters, kernel_size=1,stride=1)
        self.conv_for_score = nn.Conv2d(filters, 1, kernel_size=3,  stride=1, padding=1)
        self.score = nn.Softmax(dim=2) #(N,C,L) == (N,1,L)沿著L軸做softmax取得每個時間點上的分數 
        self.conv3 = nn.Conv2d(filters, filters//16, kernel_size=3,stride=1, padding=1)
        self.layernorm = nn.LayerNorm([filters//16, img_shape[0], img_shape[1]])
        self.swish2 = Swish()
        self.conv4 = nn.Conv2d(filters//16, in_ch, kernel_size=3,stride=1, padding=1)

    def forward(self, x):
        x1 = self.batchnorm1(self.conv1(x))
        batch, ch, h, w = x1.shape
        x1 = x1.view(batch,ch,-1)
        x1 = self.swish1(x1)
        x1 = x1.view(batch,ch,h,w)
        # print(f'x1:{x1.shape}')
        x_add = self.batchnorm2(self.conv2(x1)) + self.conv_1x1_1(x)
        # print(f'x_add:{x_add.shape}')
        score = self.score(self.conv_for_score(x_add))
        # print(f'score:{score.shape}')
        feature_map =torch.mul(score, x_add)
        # print(f'feature_map :{feature_map.shape}')
        out = self.layernorm(self.conv3(feature_map))
        batch, ch, h, w = out.shape
        out = out.view(batch,ch,-1)
        out = self.swish2(out)
        out = out.view(batch, ch, h, w)
        # print(f'out :{out.shape}')
        out = self.conv4(out) + x

        return out
    
class SPNet(nn.Module):
    def __init__(self, in_ch:int,num_class:int, img_shape:tuple, filters:int=64):
        """img_shape (h,w)"""
        super(SPNet,self).__init__()
        # model structure
        self.conv1 = nn.Conv2d(in_ch, filters, kernel_size=5,stride=1,padding=2)
        self.batchnorm1 = nn.BatchNorm2d(filters)
        self.swish1 = Swish()
        self.residualGCblock = Risudual_GC_Module(filters,filters=filters//2,img_shape = img_shape)
        self.batchnorm2 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, 1,kernel_size=3,stride=1,padding=1)
        self.score = nn.Sigmoid()
        self.globalmaxpooling = nn.AdaptiveMaxPool2d((1,1))
        self.globalavgpooling = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.5)
        self.clr = FCN(in_neuron=filters, out_neuron=num_class,num_hidden_layer=1, reduction_rate= 4)

        # utils
        self.feature_gradients = None # placeholder for the gradients

    def forward(self, x):
        x = self.batchnorm1(self.conv1(x))
        batch, ch, h, w = x.shape
        x = x.view(batch,ch,-1)
        x = self.swish1(x)
        x = x.view(batch, ch, h, w)
        h = self.residualGCblock(x)
        handdle = h.register_hook(self.activations_hook) #將h.grad 這個屬性存進 gradients
        h = self.batchnorm2(h)
        score = self.score(self.conv2(h))
        add_t = torch.mul(score,h) + h
        # print(f'shape of add_score: {add_t.shape}')
        # add_1d = add_t.view(x.shape[0],-1) #x.shape[0] 為 batchsize的數量
        gmp = self.globalmaxpooling(add_t)
        # print(f'shape of gmp: {gmp.shape}')
        gap = self.globalavgpooling(add_t)
        # print(f'shape of gap: {gap.shape}')
        features = self.dropout(torch.mul(gmp,gap))
        features = features.squeeze() #去除L軸
        # print(f'shape of out: {out.shape}')
        logits = self.clr(features)
        
        return logits
    
    def activations_hook(self, grad):
        '''hook function for getting the gradients of all feature maps'''
        self.feature_gradients = grad

    def get_feature_map_gradient(self):
        '''method for the gradient extraction'''
        return self.feature_gradients
    
    def get_feature_map(self, x):
        '''method for the featuremap exctraction :
        the current the feature maps are the output of residualGCblock module'''
        x = self.swish1(self.batchnorm1(self.conv1(x)))
        feature_maps = self.residualGCblock(x)
        return feature_maps
    
############### --Devider-- ################

class Risudual_GC_Module_for_resgcnet(nn.Module):
    def __init__(self,in_ch:int,filters:int,img_shape:tuple):
        '''Risudual_GC_Module_for_ResGCnet
           img_shape (h,w)'''
        super(Risudual_GC_Module_for_resgcnet,self).__init__()

        self.conv1 = nn.Conv2d(in_ch, filters,kernel_size=3,stride=1,padding=1)
        self.layernorm1 = nn.LayerNorm([filters, img_shape[0], img_shape[1]])
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3,stride=1, padding=1)
        self.layernorm2 = nn.LayerNorm([filters, img_shape[0], img_shape[1]])
        self.conv_1x1_1 = nn.Conv2d(in_ch, filters, kernel_size=1,stride=1)
        self.conv_for_score = nn.Conv2d(filters, 1, kernel_size=3,  stride=1, padding=1)
        self.score = nn.Softmax(dim=1) #(N,C,H,W) == (N,C,H.W)沿著C軸做softmax取得每個特徵圖上的分數 
        self.conv3 = nn.Conv2d(filters, filters//16, kernel_size=3,stride=1, padding=1)
        self.layernorm3 = nn.LayerNorm([filters//16, img_shape[0], img_shape[1]])
        self.conv4 = nn.Conv2d(filters//16, in_ch, kernel_size=3,stride=1, padding=1)

        # utils
        self.feature_gradients = None # placeholder for the gradients

    def forward(self, x):
        x1 = F.gelu(self.layernorm1(self.conv1(x)))
        # print(f'x1:{x1.shape}')
        x_add = self.layernorm2(self.conv2(x1)) + self.conv_1x1_1(x)
        # print(f'x_add:{x_add.shape}')
        score = self.score(self.conv_for_score(x_add))
        # print(f'score:{score.shape}')
        feature_map =torch.mul(score, x_add)
        # print(f'feature_map :{feature_map.shape}')
        out = F.gelu(self.layernorm3(self.conv3(feature_map)))
        # print(f'out :{out.shape}')
        out = self.conv4(out) + x

        return out
    
class ResGCNet(nn.Module):
    '''相較於SPnet，這個網路只是將BatchNormalization替換成LayerNormalization，activation finction 從swish換成gelu'''
    def __init__(self, in_ch:int,num_class:int, img_shape:tuple, filters:int=64):
        super(ResGCNet,self).__init__()
        self.conv1 = nn.Conv2d(in_ch, filters, kernel_size=5,stride=1,padding=2)
        self.layernorm1 = nn.LayerNorm([filters, img_shape[0], img_shape[1]])
        self.residualGCblock = Risudual_GC_Module_for_resgcnet(filters,filters=filters//2,img_shape = img_shape)
        self.layernorm2 = nn.LayerNorm([filters,img_shape[0], img_shape[1]])
        self.conv2 = nn.Conv2d(filters, 1,kernel_size=3,stride=1,padding=1)
        self.score = nn.Sigmoid()
        self.globalmaxpooling = nn.AdaptiveMaxPool2d(1)
        self.globalavgpooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.clr = FCN(in_neuron=filters, out_neuron=num_class,num_hidden_layer=1, reduction_rate= 4)

    def forward(self, x):
        x = F.gelu(self.layernorm1(self.conv1(x)))
        h = self.residualGCblock(x)
        handdle = h.register_hook(self.activations_hook) #將h.grad 這個屬性存進 gradients
        h = self.layernorm2(h)
        score = self.score(self.conv2(h))
        add_t = torch.mul(score,h) + h
        # print(f'shape of add_score: {add_t.shape}')
        # add_1d = add_t.view(x.shape[0],-1) #x.shape[0] 為 batchsize的數量
        gmp = self.globalmaxpooling(add_t)
        # print(f'shape of gmp: {gmp.shape}')
        gap = self.globalavgpooling(add_t)
        # print(f'shape of gap: {gap.shape}')
        features = self.dropout(torch.mul(gmp,gap))
        features = features.squeeze() #去除L軸
        # print(f'shape of out: {features.shape}')
        logits = self.clr(features)

        return logits
    
    def activations_hook(self, grad):
        '''hook function for getting the gradients of all feature maps'''
        self.feature_gradients = grad

    def get_feature_map_gradient(self):
        '''method for the gradient extraction'''
        return self.feature_gradients
    
    def get_feature_map(self, x):
        '''method for the featuremap exctraction :
        the current the feature maps are the output of residualGCblock module'''
        x = F.gelu(self.layernorm1(self.conv1(x)))
        feature_maps = self.residualGCblock(x)
        return feature_maps

class CrossEntropy(nn.Module):
    '''此類的instance使用時所輸入的第一個參數為logit(輸出層未經過activate function)，第二個參數可以是二軸(N,1)或一軸向量(N)的label'''
    def __init__(self,device):
        super(CrossEntropy,self).__init__()
        self.divice = device

    def forward(self, logits, labels):
        index = labels.to(dtype=torch.int64) #原本的label是index向量
        labels = torch.zeros(logits.size(),dtype=torch.float32).to(self.divice)
        # labels = labels.scatter_(dim=1, index=index, src=torch.ones(index.shape[0],dtype=torch.float32)) #index out of range 找不出原因
        for ndx, i in enumerate(index.squeeze()):
            labels[ndx,i] = 1.0
        # print(labels)
        log_prob = torch.log(torch.softmax(logits, dim=1))
        cross_entropy = -1 * torch.sum(labels*log_prob) / labels.shape[0]

        return cross_entropy


    
if __name__ == '__main__':
    SP_model = ResGCNet(in_ch=3,num_class=2,filters=32,img_shape=(224,224))
    # ResGCmodel = ResGCNet(in_ch=6,num_class=2,filters=32,length_of_data=360)
    dummpy_input = torch.randn(1,3,224,224)

    # torch.onnx.export(SP_model, dummpy_input, 'SP_net.onnx', opset_version=11)

    print(SP_model)
    # # print(ResGCmodel)
    batch, input_dim, h,w = 45, 3, 224,224
    tensor = torch.randn(batch, input_dim, h,w)
    logits = SP_model(tensor)
    print(logits.shape)
    print(logits)
    

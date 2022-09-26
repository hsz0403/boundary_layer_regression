import torch 
import numpy as np
def get_cost(y):#[2,Y,X] to [Y,X]
    out=torch.tensor(y.shape[1:])
    out[:]=torch.abs(y[1][:]-y[0][:])
    out=torch.softmax(out,dim=0)
    return out

def find_path(cost):#[Y,X] image to [Y,X] path
    path=np.zeros(cost.shape)
    path_image=np.zeros(cost.shape)
    total_cost=np.ones([cost.shape[0]+2,cost.shape[1]+1])*5000
    total_cost[:,0]=0
    for j in range(cost.shape[1]):
        for i in range(cost.shape[0]):
        
            
            
            total_cost[i+1][j+1]=min([total_cost[i][j],total_cost[i+1][j],total_cost[i+2][j]])+cost[i][j]
            path[i][j]=np.argmin([total_cost[i][j],total_cost[i+1][j],total_cost[i+2][j]])-1
            
    
    
    final_y=np.argmin(total_cost[:,total_cost.shape[1]-1])-1
    
    path_image[final_y,path_image.shape[1]-1]=1
    
    for i in reversed(range(path_image.shape[1]-1)):
        
        path_image[int(final_y+path[final_y,i+1]),i]=1
        
        
        final_y+=int(path[final_y,i+1])
        
    return path_image


if __name__ == '__main__':
    cost=torch.tensor([[1,2,6],[-0.5,-0.3,2],[-2,1,3],[4,5,3]])
    
    print(find_path(cost))
    

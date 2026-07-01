import torch

def PrintTensor(t):
    print("storage_offset:", t.storage_offset())
    print("is_contiguous:", t.is_contiguous())
    print("is_sparse:", t.is_sparse)
    print("shape:", t.shape)
    print("stride:", t.stride())
    print(t)
    print("\n")

def PrepareTestTensor():
    x = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]])
    PrintTensor(x)
    return x
def PrepareTestTensor2():
    x = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    PrintTensor(x)
    return x

def TestPermuteAndTranspose():
    tensor1 = PrepareTestTensor();
    
    print("Transpose for dim 0 and 1")
    tensor2 = tensor1.transpose(0, 1)
    PrintTensor(tensor2)
    
    print("Permute for {0, 2, 1}")
    tensor3 = tensor2.permute(0, 2, 1)
    PrintTensor(tensor3)
def TestContiguous():
    tensor1 = PrepareTestTensor()
    
    print("Transpose for dim 2 and 1")
    tensor1.transpose_(2, 1)
    PrintTensor(tensor1)
    
    print("Contiguous")
    tensor2 = tensor1.contiguous()
    PrintTensor(tensor2)
def TestView():
    tensor1 = PrepareTestTensor()
    print("View for {3, 8}")
    tensor2 = tensor1.view(3, 8)
    PrintTensor(tensor2)
    
    print("Transpose for dim 0 and 1")
    tensor2.transpose_(0, 1)
    PrintTensor(tensor2)
    
    print("View for {6, 4}")
    tensor3 = tensor2.view(6, 4)
    PrintTensor(tensor3)
def TestSqueezeAndUnsqueeze():
    tensor1 = PrepareTestTensor()
    print("Unsqueeze for dim 1")
    tensor2 = tensor1.unsqueeze(1)
    PrintTensor(tensor2)
    
    print("Squeeze for dim 1")
    tensor3 = tensor2.squeeze(1)
    PrintTensor(tensor3)
    
    print("Squeeze..\Libs")
    tensor4 = tensor2.squeeze()
    PrintTensor(tensor4)
def TestExpand():
    tensor1 = PrepareTestTensor2()
    print("Unsqueeze for dim 1")
    tensor2 = tensor1.unsqueeze(1)
    PrintTensor(tensor2)
    
    print("Expand for {2, 2, 2, 3} from {2, 1, 2, 3}")
    tensor3 = tensor2.expand(2, 2, 2, 3)
    PrintTensor(tensor3)
    
    print("Expand for {3, 2, 2, 3} from {2, 2, 3}")
    tensor4 = tensor1.expand(3, 2, 2, 3 )
    PrintTensor(tensor4)    
def TestRepeat():
    tensor1 = PrepareTestTensor2()
    
    print("transpose for dim 1 and 2")
    tensor2 = tensor1.transpose(1,2)
    PrintTensor(tensor2)
    
    print("Repeat for {1, 2, 1}")
    tensor3 = tensor2.repeat(1, 2, 1)
    PrintTensor(tensor2)
    PrintTensor(tensor3)
    
    
def TestNarrow():
    tensor1 = PrepareTestTensor()
    print("Narrow for dim 1, start 1, length 2, from {2, 3, 4}")
    tensor2 = tensor1.narrow(1, 1, 2)
    PrintTensor(tensor2)
    
    print("Narrow for dim 0, start 0, length 1 from {2, 2, 4}")
    tensor3 = tensor2.narrow(0, 0, 1)
    PrintTensor(tensor3)
    
    print("Narrow for dim {1, 0}, start {1, 0}, length {2, 1}, from {2, 3, 4}")
    tensor4 = tensor1.narrow(1, 1, 2).narrow(0, 0, 1)
    PrintTensor(tensor4)
def TestFlip():
    tensor1 = PrepareTestTensor()
    
    print("Flip dim 1")
    tensor2 = tensor1.flip(1)
    PrintTensor(tensor2)
    
def TestNarrowAndClone():
    tensor1 = PrepareTestTensor()
    
    print("Narrow for dim 0, start 1, length 1, from {2, 3, 4}")
    tensor2 = tensor1.narrow(0, 1, 1)
    PrintTensor(tensor2)
    
    print("Clone")
    tensor3 = tensor2.clone()
    PrintTensor(tensor3)
    
    print("Narrow for dim 1, start 0, length 2, from {2, 3, 4}")
    tensor4 = tensor1.narrow(1, 0, 2)
    PrintTensor(tensor4)
    
    print("Clone")
    tensor5 = tensor4.clone()
    PrintTensor(tensor5)
def TestTransposeAndClone():
    tensor1 = PrepareTestTensor()
    
    print("transpose for dim 1 and 2")
    tensor2 = tensor1.transpose(1, 2)
    PrintTensor(tensor2)
    
    print("Clone")
    tensor3 = tensor2.clone()
    PrintTensor(tensor3)
def TestSliceAndClone():
    tensor1 = PrepareTestTensor()
    
    print("Slice for [:,:,0:2]")
    tensor2 = tensor1[:,:,0:2]
    PrintTensor(tensor2)
    
    print("Clone")
    tensor3 = tensor2.clone()
    PrintTensor(tensor3)
def TestNarrowTransposeAndClone():
    tensor1 = PrepareTestTensor()
    
    print("Narrow for dim 1, start 1, length 2, from {2, 3, 4}")
    tensor2 = tensor1.narrow(1, 1, 2)
    PrintTensor(tensor2)
    
    print("Narrow for dim 2, start 1, length 2, from {2, 2, 4}")
    tensor3 = tensor2.narrow(2, 1, 2)
    PrintTensor(tensor3)
    
    print("transpose for dim 1 and 2")
    tensor4 = tensor3.transpose(1, 2)
    PrintTensor(tensor4)
    
    print("Clone")
    tensor5 = tensor4.clone()
    PrintTensor(tensor5)
def TestExpandAndClone():
    tensor1 = PrepareTestTensor2()
    
    print("Expand for {2, 2, 2, 3} from {2, 2, 3}")
    tensor2 = tensor1.expand(2, 2, 2, 3)
    PrintTensor(tensor2)
    
    print("Clone")
    tensor3 = tensor2.clone()
    PrintTensor(tensor3)

if __name__ == "__main__":
    #TestPermuteAndTranspose()
    #TestContiguous()
    #TestView()
    #TestSqueezeAndUnsqueeze()
    #TestExpand()
    TestRepeat()
    #TestNarrow()
    #TestFlip()
    #TestNarrowAndClone()
    #TestTransposeAndClone()
    #TestSliceAndClone()
    #TestNarrowTransposeAndClone()
    #TestExpandAndClone()


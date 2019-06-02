from models import *
import warnings


warnings.filterwarnings('ignore')

#Basic Setting
EPOCHS = 120
INIT_LR = 1e-4
BS = 16
CLASS_NUM = 2

#Loading data and augment it
trainX = load_image(train_path_x)
testX = load_image(test_path_x)
trainY,masks_trainY = load_mask(train_path_y)
testY,masks_testY = load_mask(test_path_y)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")

##========================================STEP ONE================================================##
# =================================Tumor Determine========================================#
model0=get_model()
mypotim = Adam(lr=INIT_LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model0.compile(loss='binary_crossentropy',
              optimizer=mypotim,
              metrics=['accuracy'])
print("[INFO] training classifier...")
X_train,X_valid,y_train,y_valid = train_test_split(trainX,trainY,test_size=0.1,random_state=0)
Cla = model0.fit_generator(aug.flow(X_train, y_train, batch_size=BS),
                        validation_data=(X_valid,y_valid), steps_per_epoch=len(X_train) // BS,
                        epochs=EPOCHS, verbose=1)
plot_model(model0, to_file='./results/Cla.png')

# 保存分类结果
print("[INFO] serializing classifier...")
model0.save('./results/cla_model.h5')
results0= model0.predict(testX,batch_size=len(testX),verbose=1)
results0[results0>=0.5]=1
results0[results0<0.5]=0
for i in range(1,len(results0)-1):
    if (results0[i-1,1]==1 and results0[i+1,1]==1 and results0[i,1]==0):
        results0[i,1]=1
res=pd.DataFrame(results0[:,1])
res.to_csv('./results/tumor_predict.csv')

##=================================================================================##


##====================================STEP TWO=============================================##

EPOCHS = 100
BS = 8

##Segment Tumor
X_arr = Seg_data(trainX,trainY[:,1])
y_arr = Seg_data(masks_trainY,trainY[:,1])
del trainX,trainY
print("[INFO] training classifier...")

X_train_new,y_train_new = elastic_data(X_arr,y_arr)
X_train = np.concatenate((X_arr,X_train_new))
y_train=np.concatenate((y_arr,y_train_new))
model1 = unet(Adam(lr=1e-4),(256,256,1),tversky_loss)
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
Seg = model1.fit(X_train, y_train, batch_size=BS,validation_split=0.1, 
                        epochs=EPOCHS, verbose=1,callbacks=[model_checkpoint])

plot_model(model1, to_file='./results/Seg.png')

# 保存分割结果
results1=[]
for i in range(len(results0)):
    if results0[i,1]==0:
        results1.append(np.zeros((256,256)))
    else:
        result = model1.predict(testX[i][np.newaxis,:,:,:],1,0)
        results1.append(result[0,:,:,0])
save_result('./data/Cut/test/pred',results1)

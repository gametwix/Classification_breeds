# Отчёт по работе

## Задание 1: Классификация Pet Faces

### Модель

После скачивания, считывание и нормализации данных, я перешел к созданию подели на PyTorch.

Я создал модель из 4 сверточных слоев, которые сворачивают изображение (50,50,3) до 256 фич, а дальше по этим фичам линейный слой определяет класс

```
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Sequential(nn.Conv2d(3, 32, 5),nn.ReLU(), nn.MaxPool2d(kernel_size=2,stride = 2)) 
        self.l2 = nn.Sequential( nn.Conv2d(32, 64,5),nn.ReLU(), nn.MaxPool2d(kernel_size=2,stride = 2))
        self.l3 = nn.Sequential( nn.Conv2d(64, 128,4),nn.ReLU(), nn.MaxPool2d(kernel_size=2,stride = 2))
        self.l4 = nn.Conv2d(128,256, 3)
        self.l5 = nn.Flatten()
        self.l6 = nn.Linear(256, 35)
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        return x
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr= 0.001)
```

После 150 эпох тренеровки сети мы получили данный результат.

### График обучения

![](faces_graph.png)

### Точность

```
Classification accuracy: 0.48522550544323484
Classification cat and dog accuracy: 0.9035769828926905
```

### Сonfusion matrix для по породам

![](faces_conf_mat.png)

### Сonfusion matrix для определения "Кошки против собак"

![](faces_conf_dc.png)

### Top-3 accuracy

```
Accuracy cat_Siamese: 0.8461538461538461
Accuracy dog_pomeranian: 0.8461538461538461
Accuracy cat_Bombay: 0.782608695652174
```

## Задание 2: Классификация полных изображений с помощью transfer learning

## VGG

Скачаем претренерованную модель vgg19 без линейных слоев, отключим тренировку сверточных и создадим свой линейный словй для 37 классов.

```
vgg = keras.applications.vgg19.VGG19(weights = 'imagenet', include_top = False,input_shape=(224,224,3))
for i in vgg.layers:
    i.trainable = False

model = keras.models.Sequential()
model.add(tf.keras.layers.Lambda(tf.keras.applications.vgg19.preprocess_input)) 
model.add(vgg)
model.add(Flatten())
model.add(Dense(37,activation='softmax'))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
```

После 5 эпох мы получили точность:

```
Train accuracy: 0.981058657169342
Test accuracy: 0.7941774129867554
```

### График обучения

![](vgg_graph.png)

### Сonfusion matrix

![](vgg_conf_mat.png)

### Точность отличия кошек от собак

```
Classification cat and dog accuracy vgg: 0.9783344617467841
```

![](vgg_conf_dc.png)

### Top-5 accuracy

```
Accuracy pug: 1.0
Accuracy Egyptian_Mau: 0.975
Accuracy Bombay: 0.95
Accuracy newfoundland: 0.95
Accuracy keeshond: 0.925
```

## ResNet

Попробуем сделать аналогичную модель используя претренерованную модель ResNet

```
resnet = keras.applications.resnet.ResNet152(weights = 'imagenet', include_top = False,input_shape=(224,224,3))
for i in resnet.layers:
    i.trainable = False

model2 = keras.models.Sequential()
model2.add(tf.keras.layers.Lambda(tf.keras.applications.resnet.preprocess_input)) 
model2.add(resnet)
model2.add(Flatten())
model2.add(Dense(37,activation='softmax'))
model2.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
```

После 5 эпох обучения
```
Train accuracy: 0.9878234267234802
Test accuracy: 0.8733919858932495
```

### График обучения

![](resnet_graph.png)

### Сonfusion matrix

![](resnet_conf_mat.png)

### Точность отличия кошек от собак

```
Classification cat and dog accuracy vgg: 0.992552471225457
```

![](resnet_conf_dc.png)

### Top-5 accuracy

```
Accuracy Birman: 1.0
Accuracy Bombay: 1.0
Accuracy Sphynx: 1.0
Accuracy pug: 1.0
Accuracy Persian: 0.975
```

# Вывод

Как мы в можем заметить используя претренерованные сети, в задачах в которых это возможно, можно получить намного более хороший результат за меньшее количество времени.
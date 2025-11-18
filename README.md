# PENDIENTES
 - [x] Guardar pesos del modelo durante entrenamiento
 - [ ] Agregar jitter en transforms y ver si mejora
 - [ ] Post procesado de mascaras por si quedan con ruido o hueco
 - [x] Nuevo umbral sigmoid
 - [ ] Probar learning rate scheduler
 - [ ] (!!) Arreglar funcion de submission para que funcione bien para batch_size > 1. Actualmente hay que correr el dataloader de kaggle con batch_size=1 antes de llamar a la funcion de la submission para que retorne bien la cantidad de filas.
 - [ ] Investigar efecto de preprocesado de imagenes, mas alla del data augmentation
 
# INFO

### Guardar pesos del modelo

Se guardan en el train() -> pasar por parametro nombre de archivo que deje claro cual era la arquitectura

Para restaurar:

    model = UNet(n_class=1, padding=1).to(DEVICE) #crear modelo con misma arquitectura
    
    checkpoint = torch.load("best_model.pt", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

Para reanudar entrenamiento:

    model = UNet(n_class=1, padding=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    checkpoint = torch.load("best_model.pt", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1

### Clases y funciones
class SegDataset: Para vincular imagen/mascara

def load_image: 
carga la imagen, la pasa a blanco y negro o RGB segun corresponda, y devuelve el tensor

class TestSegmentationDataset: 
como SegDataset pero solo para la carpeta TEST y NO busca mascaras (porque no hay)

def get_seg_dataloaders: 
 * toma el dataset completo con SegDataset
 * genera train_ds , val_ds, test_ds : a partir de SegDataset
 * genera test_ds_kaggle : a partir de TestSegmentationDataset
 * devuelve los data loaders

def center_crop: lo usa UNet

def model_segmentation_report: 
evalua el modelo + calcula y muestra metricas. Hay que tener cuidado con los tamaños, porque si no tenemos padding, la red devuelve imagenes mas chicas que la mascara y para poder compararlas tienen que coincidir.

### Links

UNet con arquitectura modificada:
* https://iopscience.iop.org/article/10.1088/1742-6596/1815/1/012018/pdf

UNet + attention: 
* https://arxiv.org/pdf/1804.03999
* https://www.kaggle.com/code/utkarshsaxenadn/person-segmentation-attention-unet ---> en esta muestran predicted mask vs true mask durante training!!!

UNet for human segmentation, con arquitectura original: 
* https://towardsdev.com/human-segmentation-using-u-net-with-source-code-easiest-way-f78be6e238f9

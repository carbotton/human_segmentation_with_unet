# PENDIENTES
 - [ ] Guardar pesos del modelo durante entrenamiento
 - [ ] Agregar jitter en transforms y ver si mejora
 - [ ] 
 
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

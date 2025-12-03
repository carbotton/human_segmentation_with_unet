# Segmentación de Personas con U-Net 

Este proyecto aborda la **segmentación binaria de personas** en imágenes utilizando redes convolucionales del tipo **U-Net** y variantes más avanzadas. Se trabajó con el dataset provisto en la materia Taller de Deep Learning (derivado de un desafío de Kaggle), donde cada imagen contiene una persona cuya máscara debe ser predicha píxel a píxel.

El objetivo final es obtener un modelo capaz de generar segmentaciones robustas para el set de test y producir un `submission.csv` compatible con Kaggle.

---

## 1. Motivación

La segmentación de personas es un problema clásico donde:

* Hay gran variabilidad en iluminación, tamaño de la persona, pose, fondo y color.
* El dataset contiene imágenes de **800×800**, que por limitaciones de hardware se redujeron a **256×256**.
* Existen muchas imágenes difíciles: personas muy chicas, recortadas, en blanco y negro o con fuerte brillo.

Este contexto motiva explorar **arquitecturas simples pero bien regularizadas**, así como evaluar técnicas de **data augmentation**, **postprocesamiento** y varias **configuraciones de loss**.

---

## 2. Flujo general del proyecto

El notebook sigue una narrativa secuencial:

1. **Carga y preprocesamiento de datos**

   * Normalización para RGB y B/N.
   * Máscaras convertidas a valores enteros (0/1).
   * Redimensionamiento a 256×256.

2. **Construcción del dataset y dataloaders**

   * División en train / val.
   * Transformaciones base + augmentations moderadas.

3. **Arquitecturas evaluadas**

   * **U-Net:** implementada tal y como se ve en el paper de UNet.
   * **U-Net mejorada:** padding=1, entrada RGB, BatchNorm, Dropout...
   * **Variantes:** UNet++, UNet Attention, Residual U-Net, LeakyReLU aggressive.

4. **Loss functions**

   * BCEWithLogits
   * BCEWithLogits + Dice
   * Combined loss ponderada
   * Combined loss edge
   * Focal Tversky 

---

## 3. Principales hallazgos

### 3.1 UNet base → rendimiento insuficiente

La U-Net clásica con entrada en B/N y padding=0 produce un Dice bajo (~0.41). Sirve solo como baseline.

### 3.2 Aportes clave a la mejora

Las decisiones que más impacto tuvieron fueron:

* **Entrada RGB:** mejora notable respecto a usar B/N.
* **Padding = 1:** mantiene la resolución entre capas y mejora bordes.
* **BatchNorm:** estabiliza el entrenamiento.
* **Dropout:** reduce sobreajuste.
* **Máscara en formato entero (0/1):** evita inconsistencias en la loss.
* **Combined loss:** mejor balance entre precisión y recall.

---

## 4. Mejor modelo obtenido

**Arquitectura:** UNet con bloques residuales, dropout y bloques de doble convolución.
**Loss:** combined_loss_edge
**Tamaño entrada:** 256×256
**Dice en validación:** 0.95
**Score Kaggle:** 0.93

---

## 5. Postprocesado

Se aplicó un proceso ligero para:

* Eliminar componentes pequeñas (<50 px).
* Suavizar ruido aislado.

Aunque el impacto fue moderado, contribuyó a estabilizar el score en imágenes complejas.

---

## 6. Generación del submission

El proyecto incluye funciones para:

* Ejecutar inferencia sobre todo el set de test.
* Aplicar postprocesado opcional.
* Convertir máscaras a RLE.
* Guardar el archivo `submission_YYYYMMDD.csv`.

---

## 7. Estructura del repositorio

```
.
├── Obligatorio-letra-main.ipynb   # Notebook principal con la historia completa
├── models/                        # Pesos .pth de los modelos entrenados
├── utils.py                       # Helpers, funciones auxiliares y post-procesado
├── submissions/                   # Generados desde el notebook
├── README.md  
├── config/kaggle.json
├── assets/                    
```

---

## 8. Conclusiones

* La segmentación de personas requiere una combinación de **buen preprocesamiento**, **una arquitectura sólida**, y **regularización adecuada**.
* Las variantes avanzadas no siempre implican mejoras; entender el dataset y regularizar correctamente es a menudo más importante.
* El proyecto muestra un proceso iterativo realista: hipótesis → experimento → resultado → ajuste.
* El mejor modelo combina **simplicidad, padding correcto, RGB, BatchNorm y Dropout**, logrando buena generalización.


## 9. Links útiles

UNet con arquitectura modificada:
* https://iopscience.iop.org/article/10.1088/1742-6596/1815/1/012018/pdf

UNet + attention: 
* https://arxiv.org/pdf/1804.03999
* https://www.kaggle.com/code/utkarshsaxenadn/person-segmentation-attention-unet ---> en esta muestran predicted mask vs true mask durante training!!!

UNet for human segmentation, con arquitectura original: 
* https://towardsdev.com/human-segmentation-using-u-net-with-source-code-easiest-way-f78be6e238f9
UNet for human segmentation: 
* https://towardsdev.com/human-segmentation-using-u-net-with-source-code-easiest-way-f78be6e238f9
* https://www.kaggle.com/code/divakaivan12/human-image-segmentation-with-unet

FOCAL TVERSKY LOSS
* https://arxiv.org/pdf/1810.07842, usar esta loss en los ultimos 10/20 epoch 
* https://www.igi-global.com/pdf.aspx?tid=315756&ptid=310168&ctid=4&oa=true&isxn=9781668479315

UNet usando efficientnet-b4 pre-entrenado:
* https://www.kaggle.com/code/chihjungwang/human-segmentation-by-pytorch-iou-0-92

Repo con links a todas las variantes de UNet:
* https://github.com/ShawnBIT/UNet-family

U²Net:
* https://arxiv.org/pdf/2005.09007v1 : segmentacion de cualquier cosa

UNet++:
* https://arxiv.org/pdf/1807.10165
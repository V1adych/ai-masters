import torch
import numpy as np
import torch.nn.functional as F
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2
from PIL import Image, ImageDraw
import abc
from sklearn.metrics import roc_auc_score, mean_squared_error
import os
import pandas as pd
from torch.utils.data import Dataset


def interactive_plot_metrics(data):
    """
    Создает интерактивный график для сравнения методов по различным метрикам
    с использованием Plotly. Данные передаются в виде словаря с ключом "Метод"
    и названиями метрик.
    
    :param data: словарь, где data["Метод"] — список методов,
                 а остальные ключи — метрики, значения которых списки значений.
    """
    metrics = list(set(data.keys()) - {"Метод"})
    methods = data["Метод"]

    initial_metric = 'SSIM↑' if 'SSIM↑' in metrics else metrics[0]
    initial_values = data[initial_metric]
    max_val = max(initial_values)

    colors = ['blue', 'orange'] if len(methods) == 2 else None

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=methods,
        y=initial_values,
        marker_color=colors,
        hovertemplate='%{y:.2f}',
        width=0.5
    ))

    buttons = []
    for metric in metrics:
        values = data[metric]
        max_val = max(values)
        buttons.append({
            "method": "update",
            "label": metric,
            "args": [
                {"y": [values]},  # обновляем данные
                {"title": f'Comparison of Methods Based on {metric}',
                 "yaxis": {"title": metric, "range": [0, max_val + 0.1 * max_val]}}
            ]
        })

    # Выпадающее меню и подписи осей
    fig.update_layout(
        title=f'Comparison of Methods Based on {initial_metric}',
        xaxis_title="Метод",
        yaxis_title=initial_metric,
        updatemenus=[{
            "buttons": buttons,
            "direction": "down",
            "showactive": True,
            "x": 0.5,
            "xanchor": "center",
            "y": 1.1,
            "yanchor": "top"
        }],
        template='plotly_white',
        width=500,
        height=500
    )

    fig.show()


def read_image(path, crop=None):
    """
    Читает изображение из заданного пути и преобразует его в формат RGB.

    :param path: путь к изображению.
    :param crop: координаты, по которым образется входное изображение.
    :return: изображение в формате numpy.ndarray в цветовой схеме RGB.
    """
    image = cv2.imread(str(path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if crop:
        image = image[crop]
    return image


def show_images(images, titles, grid=None, figsize=5):
    """
    Отображает список изображений с соответствующими заголовками в заданной сетке,
    используя Plotly для интерактивного зумирования и панорамирования.
    
    :param images: список изображений (numpy.ndarray) для отображения.
    :param titles: список заголовков для каждого изображения.
    :param grid: кортеж, определяющий размер сетки (строки, столбцы). По умолчанию (1, N).
    :param figsize: базовый размер фигуры (для масштабирования субплотов).
    """
    if grid is None:
        grid = (1, len(images))
    rows, cols = grid

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles)
    
    idx = 0
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            if idx < len(images):
                fig.add_trace(go.Image(z=images[idx]), row=r, col=c)
                fig.update_xaxes(showticklabels=False, row=r, col=c)
                fig.update_yaxes(showticklabels=False, row=r, col=c)
                idx += 1

    # Синхронизация осей: для каждого субплота задаем matches для x и y осей
    fig.for_each_xaxis(lambda axis: axis.update(matches='x'))
    fig.for_each_yaxis(lambda axis: axis.update(matches='y'))

    fig.update_layout(
        height=figsize * 75 * rows,
        width=figsize * 100 * cols,
        template='plotly_white',
        margin=dict(l=1, r=1, t=30, b=10),
        uirevision='constant' # чтобы снизить нагрузку на бекэнд
    )
    
    fig.show()


def preprocess_image(image: np.array, device):
    """
    Предобрабатывает изображение для подачи в patch_batch.

    :param image: исходное изображение в формате numpy.ndarray.
    :param device: устройство (CPU или GPU), на которое будет загружено изображение.
    :return: тензор изображения, готовый для модели.
    """
    image = image.astype(np.float64) / 255 * 2 - 1
    image = image.transpose(2, 0, 1)
    return torch.from_numpy(image).float().to(device)


def get_padding(side, patch_size, stride):
    subd = side - patch_size
    return int(np.ceil(subd / stride)) * stride - subd


def smart_unfold(image, patch_size, stride=None):
    """
    Разбивает изображение на блоки заданного размера с указанным шагом.

    :param image: тензор изображения.
    :param patch_size: размер патча.
    :param stride: шаг перемещения патча (если не указан, равен размеру патча).
    :return: тензор разбитых на патчи изображений.
    """
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    if stride is None:
        stride = patch_size        
    *_, h, w = image.shape
    padding = (
        get_padding(h, patch_size, stride) // 2,
        get_padding(w, patch_size, stride) // 2,
    )
    image = F.unfold(image, patch_size, stride=stride, padding=padding)
    image = image.view(1, 3, patch_size, patch_size, -1)

    return image


def get_fold_divisor(image: torch.tensor, patch_size, stride, padding):
    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    # Remove redundant samples and channels
    image = image[:1, :1]

    input_ones = torch.ones(image.shape, dtype=image.dtype)
    divisor = F.fold(F.unfold(input_ones, patch_size, stride=stride, padding=padding), image.shape[-2:], patch_size, stride=stride, padding=padding)

    return divisor


def probs_fold(probs, output_size, patch_size, stride, device):
    """
    Объединяет блоки тепловой карты.

    :param probs: тензор вероятностей для каждого патча.
    :param output_size: размер выходного изображения.
    :param patch_size: размер патча.
    :param stride: шаг перемещения патча.
    :param device: устройство (CPU или GPU).
    :return: тензор объединенной тепловой карты.
    """
    probs_repeated = probs[:, None, None].repeat(1, patch_size, patch_size, 1).view(probs.shape[0], -1, probs.shape[1])

    h, w = output_size
    padding = (
        get_padding(h, patch_size, stride) // 2,
        get_padding(w, patch_size, stride) // 2,
    )
    probs_folded = F.fold(probs_repeated, output_size, patch_size, stride=stride, padding=padding)
    divisor = get_fold_divisor(probs_folded, patch_size, stride, padding).to(device)
    return probs_folded / divisor


def patch_batch(gt, sample, patch_size, metric, device, stride=None, batch_size=1024):
    """
    Вычисляет тепловую карту метрики патчами между батчем эталонных и исследуемых изображений.

    :param gt: эталонное изображение.
    :param sample: исследуемое изображение.
    :param patch_size: размер патча.
    :param metric: функция метрики.
    :param device: устройство (CPU или GPU).
    :param stride: шаг перемещения патча (если не указан, равен размеру патча).
    :param batch_size: размер батча.
    :return: тензор тепловой карты.
    """
    if stride is None:
        stride = patch_size
    
    gt_blocks = smart_unfold(gt, patch_size, stride)
    sample_blocks = smart_unfold(sample, patch_size, stride)
    gt_blocks = torch.movedim(gt_blocks, -1, 1)[0]
    sample_blocks = torch.movedim(sample_blocks, -1, 1)[0]
    n_blocks, *shape = gt_blocks.shape

    gt_blocks = gt_blocks.view(n_blocks, *shape)
    sample_blocks = sample_blocks.view(n_blocks, *shape)

    sample_batches = torch.split(sample_blocks, split_size_or_sections=batch_size, dim=0)
    gt_batches = torch.split(gt_blocks, split_size_or_sections=batch_size, dim=0)

    with torch.no_grad():
        probs = [
        metric(
                sample_batch,
                gt_batch,
            ) for sample_batch, gt_batch in zip(sample_batches, gt_batches)
        ]
    probs = torch.cat(probs).float().to(device)
    probs = probs.view(1, -1)
    probs = probs_fold(probs, gt.shape[-2:], patch_size, stride=stride, device=device)
    probs = probs[:, 0].squeeze(0)

    return probs


def plot_interactive_heatmap(im1, im2, heatmap, label="", sr_name="SR"):
    """
    Отображает интерактивную тепловую карту поверх двух изображений.
    
    :param im1: Первое изображение (GT).
    :param im2: Второе изображение (SR).
    :param heatmap: Тепловая карта, накладываемая на изображения.
    :param label: Заголовок графика.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Image(z=im2))
    fig.add_trace(go.Image(z=im1))
    
    fig.add_trace(go.Heatmap(z=heatmap, 
                             colorscale='Reds', 
                             opacity=0.75, 
                             showscale=True))
    
    fig.data[0].visible = True
    fig.data[1].visible = False
    fig.data[2].visible = True

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        label=sr_name,
                        method="update",
                        args=[{"visible": [True, False, True]},]
                    ),
                    dict(
                        label="GT",
                        method="update",
                        args=[{"visible": [False, True, True]},]
                    )
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=1,
                xanchor="left",
                y=0,
                yanchor="top"
            )
        ],
        title=label
    )
    
    slider_steps = []
    for op in np.linspace(0, 1, 21):
        slider_steps.append(
            dict(
                method="restyle",
                args=[{"opacity": op}, [2]],
                label=str(round(op, 2))
            )
        )
    
    fig.update_layout(
        sliders=[dict(
            active=15,
            currentvalue={"prefix": "Прозрачность тепловой карты: "},
            pad={"t": 50},
            steps=slider_steps,
        )],
        width=800,
        height=600,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    fig.show()


class ArtifactDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, labled=True):
        """
        Args:
            csv_file (string): Путь к файлу с метаданными (data.csv).
            root_dir (string): Корневая директория, содержащая поддиректории sr, bic, masks.
            transform (callable, optional): Опциональные преобразования для изображений.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Извлечение имен файлов из датафрейма
        sr_name = os.path.join(self.root_dir, 'sr', f"{self.data_frame.iloc[idx]['sr_name']}.png")
        bic_name = os.path.join(self.root_dir, 'bic', f"{self.data_frame.iloc[idx]['bic_name']}.png")
        mask_name = os.path.join(self.root_dir, 'masks', f"{self.data_frame.iloc[idx]['mask_name']}.png")

        # Загрузка изображений
        sr_image = Image.open(sr_name).convert('RGB')
        bic_image = Image.open(bic_name).convert('RGB')
        mask_image = Image.open(mask_name).convert('L')  # Маска в черно-белом формате

        # Загрузка метки
        model_name = self.data_frame.iloc[idx]['model_name']
        prob = self.data_frame.iloc[idx]['artifact_prob']

        # Преобразование изображений (если задано)
        if self.transform:
            sr_image = self.transform(sr_image)
            bic_image = self.transform(bic_image)
            mask_image = self.transform(mask_image)

        sample = {
            'sr': sr_image,
            'bic': bic_image,
            'mask': mask_image,
            'model_name': model_name,
            'prob': prob
        }

        return sample


class UnlabledArtifactDataset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            sample = self.dataset[idx]
            sample['prob'] = None
            return sample
        

def prepare_contour(bi: np.ndarray, sr: np.ndarray, mask: np.ndarray):
    """
    Подготавливает контуры для сравнения двух изображений по маске.

    :param bi: первое изображение в формате numpy.ndarray.
    :param sr: второе изображение в формате numpy.ndarray.
    :param mask: маска области интереса.
    :return: два изображения с наложенными контурами.
    """
    assert bi.shape == sr.shape

    bi = cv2.resize(bi, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    bi = cv2.resize(bi, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST_EXACT)

    bi = bi.astype(np.float32)
    bi /= 255.0
    sr = sr.astype(np.float32)
    sr /= 255.0

    # Draw a border using a morphological gradient.
    k = cv2.getStructuringElement(cv2.MORPH_RECT, [3] * 2)
    border = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, k)

    shade = np.where(mask[..., None], [1.0, 1.0, 1.0], [0.5, 0.4, 0.4])

    bi *= shade
    bi = np.where(border[..., None], np.array([0.0, 0.0, 1.0])[None, None, :], bi)
    sr *= shade
    sr = np.where(border[..., None], np.array([0.0, 0.0, 1.0])[None, None, :], sr)

    bi *= 255.0
    sr *= 255.0
    bi = bi.astype(np.uint8)
    sr = sr.astype(np.uint8)

    bi = np.vstack((np.full((75, bi.shape[1], 3), 255, np.uint8), bi))
    sr = np.vstack((np.full((75, sr.shape[1], 3), 255, np.uint8), sr))

    bi = Image.fromarray(bi)
    draw = ImageDraw.Draw(bi)
    bi = np.asarray(bi)

    sr = Image.fromarray(sr)
    draw = ImageDraw.Draw(sr)
    sr = np.asarray(sr)

    return bi, sr


class BaseModel(abc.ABC):
    def __init__(self):
        super().__init__()
        self.LABELS_THRESHOLD = 0.7
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if '__get_lables__' in cls.__dict__:
            raise TypeError(f"{cls.__name__} is not allowed to override '__get_lables__'")
        if '__evaluate__' in cls.__dict__:
            raise TypeError(f"{cls.__name__} is not allowed to override '__evaluate__'")
        
    @abc.abstractmethod
    def fit(self, X_train, y_train):
        """
        Обучение модели на тренировочных данных.
        
        Args:
            X_train: Тренировочные признаки.
            y_train: Тренировочные метки.
        """
        pass

    @abc.abstractmethod
    def predict(self, X_test):
        """
        Предсказание на тестовых данных.
        
        Args:
            X_test: Тестовые признаки.
        
        Returns:
            Предсказанные значения.
        """
        pass

    def save_weights(self, path):
        """
        Сохранение весов модели в файл.
        
        Args:
            path: Путь для сохранения модели.
        """
        raise NotImplementedError("Метод сохранения не реализован!")

    def load_weights(self, path):
        """
        Загрузка весов модели из файла.
        
        Args:
            path: Путь к файлу с весами модели.
        """
        raise NotImplementedError("Метод загрузки не реализован!")
    
    def __get_lables__(self, dataset):
        """
        Получение меток датасета.
        """
        labels = []
        for i in range(len(dataset)):
            sample = dataset[i]
            labels.append(sample["prob"])
        y = np.array(labels)
        return y

    def __evaluate__(self, dataset, metric="roc_auc"):
        """
        Метод для оценки качества модели.
        """
        y_test = self.__get_lables__(dataset)
        unlabled_dataset = UnlabledArtifactDataset(dataset)
        y_pred = self.predict(unlabled_dataset)
        if metric == "roc_auc":
            y_test_binary = (y_test >= self.LABELS_THRESHOLD).astype(int)
            return roc_auc_score(y_test_binary, y_pred)
        elif metric == "mse":
            return mean_squared_error(y_test, y_pred)
        else:
            raise ValueError(f"Неизвестная метрика: {metric}")

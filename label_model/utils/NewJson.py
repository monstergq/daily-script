import cv2 as cv
import numpy as np
from .utils import *
import os, time, openslide
from geojson import dump, FeatureCollection, Feature, Polygon

    
class WriteJson:

    """
    GeoJson
    """

    def __init__(self, project_name='labelme', project_id='000001', anno_name='labelme') -> None:

        """
        初始化 GeoJson 类。

        参数：
        project_name (str): 项目名称。
        project_id (str): 项目 ID。
        cnt_type (str): 轮廓类型，默认为 'Polygon'。
        anno_name (str): 标注者名称，默认为 'labelme'。
        """

        # 信息标签
        self.info_labels = {}
        # 正确集合
        self.correct_set = (1, 1)
        # 标注者名称
        self.anno_name = anno_name
        # 项目 ID
        self.project_id = project_id
        # 项目名称
        self.project_name = project_name

        self.cnt_type = 'Polygon'

        self.info_image = None
        self.info_project = None

        # 输出
        self.header_image = None
        self.header_project = None
        self.header_features = None
        self.header_indicators = None

        # 默认
        self.image_mag = None
        self.image_type = None
        self.image_name = None

    def fill_Project(self):

        """
        填充项目信息。
        """

        self.header_project = {}
        self.header_project['project_id'] = self.project_id
        self.header_project['project_name'] = self.project_name

    def fill_Image(self, flow_path=['labelme'], create_time=None):

        """
        填充图片信息。

        参数：
        flow_path (list): 图片流程路径，默认为 ['ai']。
        create_time (str): 创建时间，默认为当前时间。

        """

        self.header_image = {}

        self.header_image['image_mag'] = self.image_mag
        self.header_image['image_name'] = self.image_name
        self.header_image['image_type'] = self.image_type
        self.header_image['image_shape'] = self.image_shape
        self.header_image['create_time'] = create_time if create_time else time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        self.header_image['image_id'] = f"{self.header_project['project_id']}{int(time.time() * 1000)}{np.random.randint(100):02d}"

        flow_path.append(self.anno_name)
        self.header_image['flow_path'] = list(set(flow_path))

    def fill_Indicators(self, indicators):

        """
        填充指标信息。

        参数：
        indicators (dict): 指标信息字典。

        indicators 样式：
        {'indicator_name': {'value': float,
                            'unit': str(),
                            'name': str()
                            },
        }
        """

        self.header_indicators = {}

        for s, e in indicators.items():

            self.header_indicators[s] = {}

            for k, v in e.items():
                self.header_indicators[s][k] = v

    def fill_Features(self, data_features, create_time=None):

        """
        填充Features数据。

        参数：
        data_features: 特征数据。
        create_time: 创建时间，默认为None。

        Features样式:
            [
                {'type': 'Feature',
                'id': str(),
                'properties': {'annotation_owner': str(0),
                               'annotation_type': str('ai'),
                               'create_time': str(),
                               'data_indicators': dict(),
                               'label_name': str(),
                               'label_color': str(),
                               'measure_type': str(),
                               'measure_relation': str(),
                               'measure_name': str(),
                               'measure_number': str()
                },
                'geometry': {'type': str(),
                            'coordinates': list()
                }
                }
            ]

        """

        self.header_features = []

        for label_name, indicators, cnt_data in data_features:

            user = '0'

            # 数据预处理
            cnt_data = np.array(cnt_data) * self.correct_set

            # 确定几何类型
            if self.cnt_type == 'Polygon':
                # 3D
                if len(cnt_data.shape) != 3:
                    assert f'错误[cnt_type]: {self.cnt_type} -- {cnt_data.shape} !'
                #
                feature = Polygon(cnt_data.tolist())
            else:
                continue

            # 构建Feature
            dict_features = Feature(
                id=f'{self.anno_name}{label_name}{int(time.time() * 1000)}{np.random.randint(100):02d}',
                properties={'annotation_owner': user,
                            'annotation_type': self.anno_name,
                            'create_time': create_time if create_time else time.strftime('%Y-%m-%d %H:%M:%S',
                                                                                            time.localtime()),
                            'data_indicators': indicators,
                            'label_name': label_name,
                            'label_color': self.dict_colors[label_name]['label_color'],
                            'label_code': self.dict_colors[label_name]['label_code'],
                            'measure_type': str(),
                            'measure_relation': str(),
                            'measure_name': str(),
                            'measure_number': str()
                            },
                geometry=feature
            )

            self.header_features.append(dict_features)

    def save_Json(self, handle_slide, features_data, version, out_dir='./'):

        """
        解析GeoJson并保存。

        参数：
        handle_slide: 图片处理对象。
        features_data: 轮廓数据。
        out_dir: 输出路径，默认为当前目录。

        """

         # 项目信息
        self.fill_Project()

        # 图片信息
        self.image_shape = str(handle_slide.level_dimensions[0])
        self.image_mag = handle_slide.properties['openslide.mpp-x'][:6]
        self.image_name, self.image_type = os.path.splitext(os.path.basename(handle_slide._filename))

        self.fill_Image()

        # 轮廓数据
        self.fill_Features(features_data)

        # 保存
        result = FeatureCollection(self.header_features)
        result['attribute'] =  {'author':"whp", 'department':"算法二组"}
        result['image'] = self.header_image
        result['project'] = self.header_project
        result['label_info'] = [{'label_name': k, 'label_color': v['label_color'], 'label_code': v['label_code']} for k, v in self.dict_colors.items()]
        result['data_indicators'] = self.header_indicators

        with open(f'{out_dir}/{self.image_name}_{version}.json', 'w', encoding='utf-8') as f:
            dump(result, f, ensure_ascii=False)

    def write_json(self, all_datas, labels, labels_dict, save_path, wsi_path, version):

        """
        主函数，处理GeoJson文件，提取信息并生成新的GeoJson文件。

        参数：
        json_path: 保存GeoJson文件路径。

        """

        # 打开WSI文件
        slide = openslide.OpenSlide(wsi_path)

        # 生成新的特征数据
        all_features = []
        self.dict_colors = labels_dict
        
        for labels_, all_datas_ in zip(labels, all_datas):

            for one_label, all_data in zip(labels_, all_datas_):

                one_cnt = all_data
                one_label = labels_dict[one_label]['label_name']
                
                one_indicator = {
                                    one_label: {
                                                    "value": 0,
                                                    "unit": str(),
                                                    "name": str()
                                                }
                                }

                # 计算面积
                one_indicator[one_label]['unit'] = 'pix^2'
                one_indicator[one_label]['name'] = f'{one_label}_面积'
                one_indicator[one_label]['value'] = cv.contourArea(one_cnt)

                all_features.append((one_label, one_indicator, [np.squeeze(one_cnt, axis=-2)]))

        print(f'the num of features: {len(all_features)}')

        # 保存新的GeoJson文件
        self.save_Json(slide, all_features, version, out_dir=save_path)
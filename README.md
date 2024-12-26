# Решение с хакатона Цифровой прорыв - ПФО, кейс Сколтех

 |  Кто | Что | Откуда |
| ------------- | ------------- | ------------- |
| Лавренченко Мария | Unet + Данные | 209М |
| Гаев Роман | Unet, backend | 209М |

## Как запусить
- docker-compose up -d --build
- http://localhost:5000/

На вход подаются только фото формата tif с 10ю каналами и размером 512x512

## Суть кейса

На вход подаются мультиспектральные снимки (изображения в 10 каналов со спутника Sentinel-2), нужно отсегментировать воду 
Датасет закрытый, просили не распространять поэтому данные по запросу.
Датасет на обучение содержит:
- Мультиспектральные снимки

Каждый снимок содержит  10 каналов.
| Name | Description                                          | Resolution |
|------|------------------------------------------------------|------------|
| B02  | Blue, 492.4 nm (S2A), 492.1 nm (S2B)                 | 10m        |
| B03  | Green, 559.8 nm (S2A), 559.0 nm (S2B)                | 10m        |
| B04  | Red, 664.6 nm (S2A), 665.0 nm (S2B)                  | 10m        |
| B05  | Vegetation red edge, 704.1 nm (S2A), 703.8 nm (S2B)  | 20m        |
| B06  | Vegetation red edge, 740.5 nm (S2A), 739.1 nm (S2B)  | 20m        |
| B07  | Vegetation red edge, 782.8 nm (S2A), 779.7 nm (S2B)  | 20m        |
| B08  | NIR, 832.8 nm (S2A), 833.0 nm (S2B)                  | 10m        | 
| B8A  | Narrow NIR, 864.7 nm (S2A), 864.0 nm (S2B)           | 20m        |
| B11  | SWIR, 1613.7 nm (S2A), 1610.4 nm (S2B)               | 20m        |
| B12  | SWIR, 2202.4 nm (S2A), 2185.7 nm (S2B)               | 20m        |

## Решение

Мы взяли Unet поскольку он из коробки поддерживает работу с мультиспектральными снимками.

На хакатоне наша итоговая метрика F1 была 0.9 

- Незатопленная область
![image](https://github.com/user-attachments/assets/64831c50-7078-4969-a066-3e0729c7e70b)
![image](https://github.com/user-attachments/assets/66ff98da-b796-45bf-a745-05a7512e11d7)


- Затопленная  область
![image](https://github.com/user-attachments/assets/bc2a79cc-9341-443a-99a3-8638494a27fe)
![image](https://github.com/user-attachments/assets/d60fd6be-0877-42ff-b139-17573c875cb1)

Если надо, доп тесты можно спарсить с https://browser.dataspace.copernicus.eu/

## Первоначальная история

Наша команда "команда" заняла 9 место на Цифровом прорыве 2024, ПФО по кейсу Сколтеха.

Команда состояла из 4х человек:

|  Кто  |  Что  |  Откуда  |
| ------------- | ------------- | ------------- |
| Лавренченко Мария | DS, Капитан | 209М |
| Гаев Роман | DS, DE | 209М |
| Панцырный Иван | Backend | 214М |
| Маркин Андрей | Backend | выпускник МАИ |

Мы не успели реализовать задуманные идеи, кейс интересный, так что хотелось бы доработать его. Проект решили полностью собрать заново и переписать бэк.

Первоначальный репозиторий: https://github.com/andrmiw9/xakaton2024

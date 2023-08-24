# ML_sim
Алгоритмы и проекты выполняемые в симуляторе ML от karpov courses

Уровни сложности: intern, junior, middle

## Similar Items Class
Класс с 3 статичными методами и возвращающей функцикцией, позволяющий по словарям с эмбедингам товаров и ценам, пределить парные товары и предсказать их новую стоимость 

similarity - Функция считает попарные похожести между всеми эмбеддингами, возвращая словарь сходств.  
knn - На вход функция принимает результат работы функции similarity, и параметр top - кол-во ближайших соседей. Она выдает словарь с парами item_id - список top ближайших товаров.  
knn_price - На вход функция принимает результат работы функции knn и словарь price с ценами для каждого товара. На выходе выдавая средневзвешенную цену top ближайших соседей.  
transform - Преобразует исходный словарь эмбеддингов в словарь с новыми ценами для всех товаров.

## Data Quality  
Автоматизация DQ с проверочными классами на полноту, актуальность, согласованность, доступность, достоверность и вывод результатов  

## SKU Uniqueness  
knn_uniqueness - Метрика уникальности товара, насколько тот или иной товар в контексте группы товаров, является «не таким, как все». Эмбеддинги построены на истории покупок (матрица Пользователи x Товары), содержат информацию о паттернах пользовательского поведения.  
kde_uniqueness - Имплементация KDE алгоритма scikit-learn.  
group_diversity - Метрика расчета разнообразия группы товаров, использует метрику уникальности на основе KDE-алгоритма.

## HOUSE PRICE PREDICTION  
Предсказание цены на будущее SKU из тестового набора данных, достигнув порогового значения метрики RMSE.

## multiprocessing_clean
предобработка текста используя multiprocessing

## Elasticity Feature
Функция на основе исторических данных оценивает эластичность (спроса по цене) для каждого SKU 

## Cumulative Gain (только с numpy)
Функция по подсчету суммы численных оценок релевантности в заданом количестве

## Discounted Cumulative Gain (только с numpy)
Функция добавляет штраф к релевантности если важный документ попал в конец списка или добавлять вес если этот документ попал в начало списка выдачи.

## Normalized Discounted Cumulative Gain (только с numpy)
Нормализованная метрика от 0 до 1. Функция делит DCG на сортированный DCG в порядке убывания(так получается максимальное значение DCG для конкретного запроса и заданного набора выдачи (параметр k) или IDCG (Ideal Discounted Cumulaive Gain)). 

## Predict price for new products
Функция по заполнению пропусков для товаров-новинок, заполняется среднем значением продаж по категориям этого товара

## sMAPE (только с numpy) 
Симметричное MAPE, где в знаменателе стоит уже сумма модулей предсказания и факта, а не только факта.

## RMSLE (только с numpy)

##Lifetime Value
Асимемтричная метрика для оценки количества денег, которое принесёт клиент за всё время, что будет пользоваться сервисом 

## Stocks
Функция для постобработки предсказаний модели ценообразования

## Abstract Selector  
Рефакторинг кода с использованием декоратора 

## Valid Emails
Функция проверяющая правильность email адреса

# Skillbox - Мастер-класс: Разведочный анализ данных + Подготовка отчетов - Kiva.org
***

## Слайды

[Skillbox - MK EDA+Reports - Kiva.org](https://docs.google.com/presentation/d/11XWrGXvFHx21Mh5fmL95uD4jh0ksUavW6ewMjZQwP88/edit?usp=sharing)

## Датасет

[Скачать датасет c kaggle.com](https://www.kaggle.com/kiva/data-science-for-good-kiva-crowdfunding/download)

Рзаместить в директории на один уровень выше директории с ноутбуком: `../kiva`

Для того, чтобы корректно работали готовые инструкции по чтению из файлов:

```python
df_kiva_loans = pd.read_csv("../kiva/kiva_loans.csv")
df_mpi = pd.read_csv("../kiva/kiva_mpi_region_locations.csv")
```

## Окружение

Можете работать в своем окружении, либо создать виртуальное:

```sh
# создать окружение
conda create -n myenv

# активировать окружение
conda activate myenv

# установить библиотеки и зависимости
conda config --add channels conda-forge
conda install jupyter jupyterlab ipykernel pandas numpy missingno plotly

# деактивировать и удалить окружение после работы
conda deactivate 
conda env remove -n myenv
```


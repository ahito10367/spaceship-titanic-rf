# Spaceship Titanic: Random Forest Pipeline

## Задача
Мультиклассификация (бинарная): предсказать, был ли пассажир телепортирован (`Transported`) во время аномалии на космическом корабле.

## Стек технологий
- **Язык**: Python 3.x
- **Обработка данных**: `pandas`, `numpy`
- **Моделирование**: `scikit-learn` (`RandomForestClassifier`, `GridSearchCV`)
- **Валидация**: 5-Fold Cross-Validation

## Пайплайн решения
1. **Feature Engineering**: извлечение палубы и борта из `Cabin` (`CabinDeck`, `CabinSide`)
2. **Обработка пропусков**: 
   - Числовые (`Age`, расходы) → медиана из обучающей выборки
   - Категориальные (`HomePlanet`, `CryoSleep`, `Destination`, `Cabin*`) → мода из обучающей выборки
3. **Кодирование**: One-Hot Encoding через `pd.get_dummies(dtype=int)`
4. **Синхронизация**: `X.align(X_test, join='inner', axis=1)` гарантирует идентичный набор признаков
5. **Обучение**: `GridSearchCV` по `n_estimators`, `max_depth`, `min_samples_split`
6. **Интерпретация**: анализ `feature_importances_` лучшей модели

## Результаты
| Метрика | Значение |
|---------|----------|
| **Best CV Accuracy** | `0.7920` |
| **Лучшие гиперпараметры** | `{'max_depth': 7, 'min_samples_split': 5, 'n_estimators': 100}` |

>  **Ключевой инсайт**: Признаки `CryoSleep`, `CabinSide` и суммарные расходы часто оказываются наиболее значимыми, что коррелирует с логикой эвакуации и распределением пассажиров по отсекам.

##  Как запустить
```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Запустить скрипт (или открыть ноутбук)
python notebooks/src/spaceship-titanic-rf.ipynb

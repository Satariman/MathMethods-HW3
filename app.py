from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

from helpers import add_predictions, classification_metrics, missing_features


APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / 'model.joblib'
CONFIG_PATH = APP_DIR / 'config.json'
TEST_PATH = APP_DIR / 'test.csv'


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_test_data():
    return pd.read_csv(TEST_PATH)


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


def threshold_point(y_true, probability, threshold):
    result = classification_metrics(y_true, probability, threshold)
    tn, fp, fn, tp = result['confusion_matrix'].ravel()
    fpr = fp / (fp + tn) if fp + tn else 0
    return fpr, result['recall'], result['precision']


if not all(path.exists() for path in [MODEL_PATH, CONFIG_PATH, TEST_PATH]):
    st.error('Не найдены файлы модели. Сначала выполните ячейку сохранения в ноутбуке.')
    st.stop()


model = load_model()
test_data = load_test_data()
features = list(model.feature_names_in_)
X_test = test_data[features]
y_test = test_data['target']
probability_test = test_proba(model, X_test)
default_threshold = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))['threshold']

st.set_page_config(page_title='Wine Quality', layout='wide')
st.title('Рекомендация вина')
threshold = st.sidebar.slider(
    'Порог хорошего вина', 0.0, 1.0, float(default_threshold), 0.001
)

single_tab, metrics_tab, batch_tab, coefficients_tab = st.tabs([
    'Одно вино', 'Метрики модели', 'Пакетный прогноз', 'Коэффициенты'
])

with single_tab:
    st.subheader('Оценка одного вина')
    with st.form('single_wine'):
        columns = st.columns(2)
        values = {}
        for index, feature in enumerate(features):
            with columns[index % 2]:
                if feature == 'is_red':
                    wine_type = st.radio('Тип вина', ['Белое', 'Красное'], horizontal=True)
                    values[feature] = int(wine_type == 'Красное')
                else:
                    values[feature] = st.number_input(
                        feature,
                        min_value=float(X_test[feature].min()),
                        max_value=float(X_test[feature].max()),
                        value=float(X_test[feature].median()),
                    )
        submitted = st.form_submit_button('Оценить')

    if submitted:
        wine = pd.DataFrame([values], columns=features)
        probability = model.predict_proba(wine)[0, 1]
        st.metric('Вероятность хорошего вина', f'{probability:.1%}')
        if probability >= threshold:
            st.success('Рекомендовать')
        else:
            st.warning('Не рекомендовать')

with metrics_tab:
    st.subheader('Метрики на тестовой выборке')
    result = classification_metrics(y_test, probability_test, threshold)
    first, second, third = st.columns(3)
    first.metric('Precision', f"{result['precision']:.3f}")
    second.metric('Recall', f"{result['recall']:.3f}")
    third.metric('F1', f"{result['f1']:.3f}")

    figure, axis = plt.subplots(figsize=(5, 4), layout='constrained')
    metrics.ConfusionMatrixDisplay(
        result['confusion_matrix'], display_labels=['Не хорошее', 'Хорошее']
    ).plot(ax=axis, cmap='Blues', colorbar=False)
    axis.grid(False)
    axis.set(xlabel='Прогноз модели', ylabel='Фактический класс')
    st.pyplot(figure, width='content')

    roc_column, pr_column = st.columns(2)
    with roc_column:
        roc_figure, roc_axis = plt.subplots(figsize=(5, 4), layout='constrained')
        fpr, tpr, _ = metrics.roc_curve(y_test, probability_test)
        point_fpr, point_tpr, point_precision = threshold_point(y_test, probability_test, threshold)
        roc_axis.plot(fpr, tpr, label='ROC')
        roc_axis.plot([0, 1], [0, 1], '--', color='gray')
        roc_axis.scatter(point_fpr, point_tpr, label=f'Порог {threshold:.3f}')
        roc_axis.set(xlabel='FPR', ylabel='TPR', title='ROC-кривая')
        roc_axis.legend()
        st.pyplot(roc_figure, width='stretch')

    with pr_column:
        pr_figure, pr_axis = plt.subplots(figsize=(5, 4), layout='constrained')
        precision, recall, _ = metrics.precision_recall_curve(y_test, probability_test)
        pr_axis.plot(recall, precision, label='PR')
        pr_axis.scatter(point_tpr, point_precision, label=f'Порог {threshold:.3f}')
        pr_axis.set(xlabel='Recall', ylabel='Precision', title='PR-кривая')
        pr_axis.legend()
        st.pyplot(pr_figure, width='stretch')
    st.info('Больший порог делает подборку меньше и точнее; меньший — шире, но с большим числом посредственных вин.')

with batch_tab:
    st.subheader('Пакетный прогноз')
    uploaded = st.file_uploader('CSV с признаками вин', type='csv')
    if uploaded is not None:
        batch = pd.read_csv(uploaded)
        missing = missing_features(batch, features)
        if missing:
            st.error(f"В файле нет столбцов: {', '.join(missing)}")
        else:
            prediction = add_predictions(model, batch, features, threshold)
            st.dataframe(prediction, hide_index=True)
            st.download_button(
                'Скачать прогноз', prediction.to_csv(index=False).encode('utf-8'),
                'wine_predictions.csv', 'text/csv'
            )

with coefficients_tab:
    st.subheader('Что выучила модель')
    coefficients = pd.Series(
        model.named_steps['logisticregression'].coef_[0], index=features
    ).sort_values()
    colors = np.where(coefficients >= 0, '#2a7fb8', '#d95f5f')
    figure, axis = plt.subplots(figsize=(7, 5), layout='constrained')
    coefficients.plot.barh(ax=axis, color=colors)
    axis.axvline(0, color='gray', linewidth=1)
    axis.set(xlabel='Коэффициент после масштабирования', ylabel='Признак')
    st.pyplot(figure, width='content')
    st.caption('Положительные коэффициенты повышают вероятность хорошего вина, отрицательные — понижают её.')

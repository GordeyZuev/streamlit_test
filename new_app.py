import streamlit as st
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
import plotly.express as px
from catboost import CatBoostClassifier
from pathlib import Path

st.set_page_config(page_title="Churn Demo", page_icon="🤖", layout="wide")

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "churn_model.cbm"

@st.cache_resource
def load_model():
    """Загружаем CatBoost‑модель из файла .cbm и возвращаем вместе со списком признаков."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Не найдена модель в папке models/")

    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    feature_names = model.feature_names_
    return model, feature_names

@st.cache_data
def load_data(uploaded_file):
    """Читаем загруженный CSV в DataFrame (кэшируем, чтобы не читать повторно)."""
    return pd.read_csv(uploaded_file)


def prepare_features(df, feature_names):
    """Приводим входные данные к формату обучения: строки для категориальных, нужный порядок столбцов."""
    df_proc = df.copy()
    for col in feature_names:
        if col in df_proc.columns:
            if df_proc[col].dtype == 'object' or df_proc[col].dtype == 'bool':
                df_proc[col] = df_proc[col].astype(str)

    missing = [col for col in feature_names if col not in df_proc.columns]
    if missing:
        raise ValueError(
            "В данных не хватает обязательных колонок: " + ", ".join(missing)
        )

    return df_proc[feature_names]


def to_numeric_churn(series: pd.Series) -> pd.Series:
    """Преобразуем поле churn в числовой вид (0/1) независимо от формата входных значений."""
    if is_bool_dtype(series):
        return series.astype(int)
    if is_numeric_dtype(series):
        return pd.to_numeric(series, errors='coerce')

    mapping = {
        'true': 1, 'yes': 1, '1': 1, 'уйдет': 1, 'ушел': 1,
        'false': 0, 'no': 0, '0': 0, 'остался': 0, 'останется': 0
    }

    lowered = series.astype(str).str.strip().str.lower()
    converted = lowered.map(mapping)
    if converted.isna().any():
        raise ValueError("Колонка 'churn' должна содержать бинарные значения (0/1/Yes/No).")
    return converted.astype(float)


try:
    MODEL, FEATURE_NAMES = load_model()
except Exception as model_error:
    st.error(f"❌ {model_error}")
    st.stop()


# --- Основной интерфейс и загрузка файла ---
st.title("🤖 Мини‑дешборд по оттоку клиентов")
st.write("Загрузите CSV и сразу получите предсказания заранее обученной CatBoost‑модели.")

uploaded_file = st.sidebar.file_uploader("Загрузите CSV с клиентами", type=["csv"])

if uploaded_file is None:
    st.info("👈 Для старта нужен файл. Можете использовать churn_clients_demo.csv")
    st.stop()


try:
    df = load_data(uploaded_file)
    if 'churn' not in df.columns:
        st.error("В данных должна быть колонка `churn`")
        st.stop()

    features = prepare_features(df, FEATURE_NAMES)
    probs = MODEL.predict_proba(features)[:, 1]
    preds = (probs >= 0.5).astype(int)

    df_result = df.copy()
    df_result['prediction'] = preds
    df_result['prob_leave'] = probs
    df_result['churn_flag'] = to_numeric_churn(df_result['churn'])
except Exception as data_error:
    st.error(f"Не удалось обработать файл: {data_error}")
    st.stop()


st.subheader("📊 Быстрый обзор")
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.metric("👥 Всего клиентов", len(df_result))

with col_b:
    churn_rate = df_result['churn_flag'].mean() * 100
    st.metric("🚪 Фактический отток", f"{churn_rate:.1f}%")

with col_c:
    pred_rate = df_result['prediction'].mean() * 100
    st.metric("🔮 Предсказанный отток", f"{pred_rate:.1f}%")


col_d, col_e, col_f = st.columns(3)

with col_d:
    avg_account = df_result.get('accountlength', pd.Series(dtype=float)).mean()
    if pd.notna(avg_account):
        st.metric("📅 С нами в среднем", f"{avg_account:.0f} мес.")
    else:
        st.metric("📅 С нами в среднем", "—")

with col_e:
    intl_share = (
        df_result.get('internationalplan', pd.Series(dtype=str))
        .astype(str)
        .str.lower()
        .isin(['yes', 'true', '1'])
        .mean()
        * 100
    )
    if not pd.isna(intl_share):
        st.metric("🌍 Международный план", f"{intl_share:.1f}% клиентов")
    else:
        st.metric("🌍 Международный план", "—")

with col_f:
    avg_support = df_result.get('customerservicecalls', pd.Series(dtype=float)).mean()
    if pd.notna(avg_support):
        st.metric("☎️ Обращений в поддержку", f"{avg_support:.1f}")
    else:
        st.metric("☎️ Обращений в поддержку", "—")


st.caption(
    "Предсказанный отток — доля клиентов, для которых модель оценила вероятность ≥ 50%."
)

st.subheader("🎚️ Фильтры для визуализаций")
state_filter = []
if 'state' in df_result.columns:
    state_filter = st.multiselect(
        "Ограничить по штатам",
        sorted(df_result['state'].astype(str).unique()),
        placeholder="Все штаты"
    )

viz_df = df_result.copy()
if state_filter:
    viz_df = viz_df[viz_df['state'].astype(str).isin(state_filter)]

if viz_df.empty:
    st.warning("По выбранным фильтрам записей нет — показываем исходные данные.")
    viz_df = df_result.copy()


st.subheader("📈 Ключевые визуализации")
chart_df = viz_df.copy()
chart_df['churn_label'] = chart_df['churn_flag'].map({0.0: "Остался", 1.0: "Ушел"})
chart_df['churn_label'] = chart_df['churn_label'].fillna("Неизвестно")

pie_fig = px.pie(chart_df, names='churn_label', title="Фактический отток", hole=0.35)
pie_fig.update_traces(textposition='inside', textinfo='percent+label')
st.plotly_chart(pie_fig, use_container_width=True)



st.markdown("**Категории vs риск оттока**")
if 'internationalplan' in viz_df.columns:
    plan_df = viz_df[['internationalplan', 'churn_flag', 'prob_leave']].copy()
    plan_df['internationalplan'] = plan_df['internationalplan'].astype(str)
    plan_summary = plan_df.groupby('internationalplan').agg(
        Фактический=('churn_flag', 'mean'),
        Средняя_вероятность=('prob_leave', 'mean')
    ).reset_index()

    show_avg_line_plan = st.checkbox("Показать среднюю вероятность модели", value=True, key="plan_avg")
    
    plan_melt = plan_summary.melt(
        id_vars='internationalplan',
        value_vars=['Фактический'],
        var_name='Метрика',
        value_name='rate'
    )
    plan_melt['rate'] = plan_melt['rate'] * 100
    bar_fig = px.bar(
        plan_melt,
        x='internationalplan',
        y='rate',
        color='Метрика',
        barmode='group',
        text='rate',
        labels={'internationalplan': 'International plan', 'rate': 'Процент, %'},
        title="Как международный план связан с оттоком"
    )
    bar_fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    
    # Добавляем линию среднего, если включен чекбокс
    if show_avg_line_plan:
        overall_mean_pred = viz_df['prob_leave'].mean() * 100
        bar_fig.add_hline(
            y=overall_mean_pred,
            line_dash="dash",
            line_color="#3498db",
            annotation_text=f"Среднее {overall_mean_pred:.1f}%",
            annotation_position="right"
        )
    
    bar_fig.update_layout(yaxis_tickformat='.0f', uniformtext_minsize=10, uniformtext_mode='hide')
    st.plotly_chart(bar_fig, use_container_width=True)
else:
    st.info("Колонка internationalplan не найдена — сравнить категории не получится.")



st.markdown("**Звонки в поддержку vs риск оттока**")
support_col = None
for col in ['customerservicecalls', 'number_customer_service_calls', 'numbervmailmessages']:
    if col in viz_df.columns:
        support_col = col
        break

if support_col:
    support_df = viz_df[[support_col, 'churn_flag', 'prob_leave']].copy()
    support_summary = support_df.groupby(support_col).agg(
        Фактический=('churn_flag', 'mean'),
        Средняя_вероятность=('prob_leave', 'mean')
    ).reset_index()
    support_summary[support_col] = support_summary[support_col].astype(int)

    show_avg_line_support = st.checkbox("Показать среднюю вероятность модели", value=True, key="support_avg")
    
    support_melt = support_summary.melt(
        id_vars=support_col,
        value_vars=['Фактический'],
        var_name='Метрика',
        value_name='rate'
    )
    support_melt['rate'] = support_melt['rate'] * 100
    support_fig = px.bar(
        support_melt,
        x=support_col,
        y='rate',
        color='Метрика',
        barmode='group',
        text='rate',
        labels={support_col: 'Количество звонков', 'rate': 'Процент, %'},
        title="Как количество звонков в поддержку связано с оттоком"
    )
    support_fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    
    # Добавляем линию среднего, если включен чекбокс
    if show_avg_line_support:
        overall_mean_pred_support = viz_df['prob_leave'].mean() * 100
        support_fig.add_hline(
            y=overall_mean_pred_support,
            line_dash="dash",
            line_color="#3498db",
            annotation_text=f"Среднее {overall_mean_pred_support:.1f}%",
            annotation_position="right"
        )
    
    support_fig.update_layout(yaxis_tickformat='.0f', uniformtext_minsize=10, uniformtext_mode='hide')
    st.plotly_chart(support_fig, use_container_width=True)
else:
    st.info("Колонка с количеством звонков в поддержку не найдена.")


st.subheader("🔍 Предсказания модели (первые 30 строк)")
st.dataframe(df_result.head(30), use_container_width=True)


st.subheader("📝 Сделать собственное предсказание")

# Автоматически определяем категориальные колонки по типу данных
categorical_cols = [col for col in FEATURE_NAMES 
                  if col in df_result.columns 
                  and (df_result[col].dtype == 'object' or df_result[col].dtype == 'bool')]

with st.form("manual_prediction"):
    col_cat, col_num = st.columns(2)

    manual_input = {}
    
    with col_cat:
        for col in categorical_cols:
            options = sorted(df_result[col].astype(str).unique().tolist())
            manual_input[col] = st.selectbox(f"{col}", options, key=f"cat_{col}")

    numeric_cols = [col for col in FEATURE_NAMES if col not in categorical_cols]
    numeric_cols = [col for col in numeric_cols if col in df_result.columns]

    with col_num:
        for col in numeric_cols:
            col_values = df_result[col]
            default = float(col_values.median())
            min_val = float(col_values.min())
            max_val = float(col_values.max())
            manual_input[col] = st.number_input(
                col,
                value=default,
                min_value=min_val,
                max_value=max_val,
                key=f"num_{col}",
            )

    submitted = st.form_submit_button("Предсказать")

if submitted:
    try:
        input_dict = {col: [val] for col, val in manual_input.items()}
        input_df = pd.DataFrame(input_dict)
        # Преобразуем категориальные колонки в строки
        for col in categorical_cols:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(str)
        input_df = input_df[FEATURE_NAMES]

        manual_prob = MODEL.predict_proba(input_df)[0][1]
        manual_pred = int(manual_prob >= 0.5)

        st.success(
            f"Предсказание: {'Уйдет' if manual_pred else 'Останется'} "
            f"(вероятность оттока {manual_prob:.1%})"
        )
    except Exception as manual_error:
        st.error(f"Не удалось сделать предсказание: {manual_error}")

st.caption("Модель обучена заранее в ноутбуке streamlit_webinar.ipynb. Здесь мы только применяем её к новым данным.")
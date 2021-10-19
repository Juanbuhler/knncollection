import random

import streamlit as st
from knncollection import knncollection
import os
import glob
import numpy as np
import cv2

page_length = 50
st.set_page_config(layout='wide')

dirs = [d.split('/')[0] for d in glob.glob('*/knnCollection.pickle')]
dir = st.sidebar.selectbox('Collection', dirs)
files = glob.glob(dir + '/*')
saved_collection = dir + '/knnCollection'
saved_model = dir + '/knnModel'


@st.cache
def get_collection():
    collection = knncollection.KnnCollection('', save_name=saved_collection)
    return collection


def main():
    collection = get_collection()
    knn_model = knncollection.KnnModel(saved_model)
    knn_labels = []
    knn_files = []
    knn_hashes = []
    for key, value in knn_model.labels.items():
        knn_files.append(key)
        knn_hashes.append(value[0])
        knn_labels.append(value[1])
    knn_labels = [''] + list(set(knn_labels))

    knn_predictions = ['']
    if knn_model.predictions is not None:
        knn_predictions = [''] + list(set(list(knn_model.predictions)))
        knn_predictions.sort()

    st.sidebar.text(str(len(collection.files)) + ' images')
    st.sidebar.markdown('---')

    similarity = st.sidebar.checkbox('Similarity')
    image_info = st.sidebar.checkbox('Image Info')
    preds = st.sidebar.checkbox('Show Predictions')
    #show_labels = st.sidebar.selectbox('Show Labels', knn_labels)
    show_labels = st.sidebar.checkbox('Show Labels')
    show_predictions = st.sidebar.selectbox('Show Predictions', knn_predictions)
    reverse_predictions = st.sidebar.checkbox('Predictions in increasing order')
    st.sidebar.markdown('---')

    label_images = st.sidebar.checkbox('Label Images')
    labelScope = st.sidebar.selectbox('Label Scope', ['train', 'test'])

    if 'search' not in st.session_state:
        st.session_state['search'] = None
        search = None
    else:
        search = st.session_state['search']

    if search is None:
        page = 1
        if not show_labels and not show_predictions:
            page = st.slider("Page", min_value=1, max_value=int(len(collection.files) / page_length))
        display_files = collection.files[page * page_length:(page + 1) * page_length]
        display_hashes = collection.hashes[page * page_length:(page + 1) * page_length]
        if knn_model.predictions is not None:
            display_predictions = knn_model.predictions[page * page_length:(page + 1) * page_length]
            display_confidences = knn_model.confidences[page * page_length:(page + 1) * page_length]

    elif 'sim_hash' in search.keys():
        s = collection.similarity_search(search['sim_hash'], num_assets=page_length)
        display_files = [collection.files[j] for j in s]
        display_hashes = [collection.hashes[j] for j in s]
        if knn_model.predictions is not None:
            display_predictions = [knn_model.predictions[j] for j in s]
            display_confidences = [knn_model.confidences[j] for j in s]
        if st.button('Reset Search'):
            st.session_state['search'] = None
            st.experimental_rerun()
        st.sidebar.markdown('---')
        n_changes = st.sidebar.slider('Hash changes', min_value=1, max_value=2000, value=1)
        amp_change = st.sidebar.slider('Change Amplitude', min_value=1, max_value=16, value=1)
        if st.sidebar.button('Perturb Hash'):
            h = np.copy(search['sim_hash'])
            for i in range(0, n_changes):
                h[random.randint(0, 2047)] += random.randint(-amp_change, amp_change)
            search = {'sim_hash': h}
            st.session_state['search'] = search
            st.experimental_rerun()


    if show_labels:
        display_files = knn_files
        display_hashes = knn_hashes
        if knn_model.predictions is not None:
            display_predictions = []
            display_confidences = []
            for f in display_files:
                index = collection.files.index(f)
                display_predictions.append(knn_model.predictions[index])
                display_confidences.append(knn_model.confidences[index])

    if show_predictions != '' and knn_model.predictions is not None:
        d_files = np.array(collection.files)[knn_model.predictions == show_predictions]
        d_hashes = np.array(collection.hashes)[knn_model.predictions == show_predictions]
        d_confidences = np.array(knn_model.confidences)[knn_model.predictions == show_predictions]
        if not reverse_predictions:
            d_confidences = -d_confidences
        sorter = np.argsort(d_confidences)
        display_files = [d_files[j] for j in sorter][:page_length]
        display_hashes = [d_hashes[j] for j in sorter][:page_length]
        display_confidences = [d_confidences[j] for j in sorter][:page_length]
        display_predictions = np.array(knn_model.predictions)[knn_model.predictions == show_predictions][:page_length]


    n_cols = 5
    cols = st.columns(n_cols)
    i_col = 0
    for i, file in enumerate(display_files):
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cols[i_col].image(img)

        if similarity:
            if cols[i_col].button("similar", key='sim'+display_files[i]):
                search = {'sim_hash': display_hashes[i]}
                st.session_state['search'] = search
                st.experimental_rerun()

        if label_images:
            if display_files[i] in knn_model.labels.keys():
                value = knn_model.labels[display_files[i]][1]
            else:
                value = ''
            l = cols[i_col].text_input('Label', key='label'+display_files[i], value=value)
            if l != '':
                knn_model.label(display_files[i], display_hashes[i], l, scope=labelScope)

        if image_info:
            cols[i_col].text(display_files[i].split('/')[-1])
            cols[i_col].text(str(img.shape[0]) + 'x' + str(img.shape[1]))

        if preds:
            if knn_model.predictions is None:
                cols[i_col].text('None')
            else:
                cols[i_col].text(display_predictions[i])
                cols[i_col].text(abs(display_confidences[i]))


        cols[i_col].markdown('---')
        i_col += 1
        if i_col >= n_cols:
            i_col = 0

    st.sidebar.markdown('---')

    n_clusters = st.sidebar.slider('Number of Clusters', min_value=5, max_value=100, value=5)
    if st.sidebar.button('Auto Group'):
        knn_model.auto_label(collection.files, collection.hashes, n_clusters=n_clusters)

    if st.sidebar.button('Train KNN Model'):
        knn_model.save_model()
        knn_model.train_apply(collection.hashes)
        st.experimental_rerun()




if __name__ == "__main__":
    main()

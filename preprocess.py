from knncollection import knncollection 
import glob

files = glob.glob('PomologicalWatercolors/*.jpg')
col = knncollection.KnnCollection(files, save_name='PomologicalWatercolors/knnCollection')
print('Collection done. Run browser with')
print('streamlit run sim_search.py')

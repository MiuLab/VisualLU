

def get_label_set(task):
    if task == 'conll2003':
        label_set = {'PER', 'LOC', 'ORG'}
    elif task == 'bc5cdr':
        label_set = {'Chemical', 'Disease'}
    elif task == 'mitmovie':
        label_set = {'person', 'title'}
    elif task == 'wnut2017':
        label_set = {'corporation', 'creative_work', 'group',
                     'location', 'person', 'product'}
        
    return label_set
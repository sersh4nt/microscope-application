from datetime import datetime
import os


class DatabaseHandler:
    def __init__(self, path):
        self.path = path
        self.records = dict()
        self.load()
        print(self.records)

    def load(self):
        for component_name in os.listdir(self.path):
            component_folder = os.path.join(self.path, component_name)
            for folder in os.listdir(component_folder):
                if folder == 'records':
                    records_folder = os.path.join(component_folder, folder)
                    for record in os.listdir(records_folder):
                        if record.lower().endswith('.txt'):
                            if component_name not in self.records:
                                self.records[component_name] = list()
                            self.records[component_name].append(DataRecord().load(record, component_name))


class DataRecord:
    def __init__(self, date=datetime.now().strftime("%d-%m-%Y"), image='', component='', number=0):
        self.date = date
        self.image = image
        self.component = component
        self.number = number

    '''directory='data/<class_name>', example: directory='data/ad620b' '''
    def save(self, class_directory):
        if not os.path.exists(os.path.join(class_directory, 'records')):
            pass

    def load(self, record='', component=''):
        return DataRecord(record[:10], record, component, int(record[11:14]))

    def __str__(self):
        return '{} {} {} {:04d}'.format(self.date, self.image, self.component, self.number)

    def __repr__(self):
        return '{} {} {} {:04d}'.format(self.date, self.image, self.component, self.number)

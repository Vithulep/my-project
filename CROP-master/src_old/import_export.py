class Import_export():
    def __init__(self,allocation,import_export):
        self.allocation=allocation
        self.import_export=import_export
    def do_import_export(self):
        # print(self.allocation)
        for district_id in self.allocation.keys():
            for crop_id in self.allocation[district_id].keys():
                # print(crop_id)
                self.allocation[district_id][crop_id]=self.allocation[district_id][crop_id]+self.import_export[district_id][crop_id]
                
import os

class Counter:

    def __init__(self, file):
        self.file = file
        self.counter = 0

        self._read_counter(file)

    def _read_counter(self, file):
        basename = os.path.basename(file).split('.')[0]
        if not os.path.exists(file):
            fw = open(file, 'w')
            fw.write(f'{basename}={0}')
            fw.write('\n')
            fw.close()
            return

        with open(file, 'r') as fr:
            for line in fr:
                bname, counter = line.strip().split('=')
                if basename == bname:
                    self.counter = int(counter)
            fr.close()

    def write(self):
        file = self.file
        with open(file, '')
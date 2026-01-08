from src import PyProtein
import sys, time

PDBID = sys.argv[1].encode('utf-8')
protein = PyProtein(PDBID)
if not protein.loadPQRFile(PDBID+'.pqr'.encode('utf-8')):
    sys.exit('read PQR file failed')
s_time = time.time()
protein.atomwise_interaction(10., 40.)
e_time = time.time()
print(e_time-s_time)
print(protein.feature_FRIs())
protein.Deallocate()

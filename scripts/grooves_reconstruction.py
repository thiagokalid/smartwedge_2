import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import matplotlib

import matplotlib.pyplot as plt
import numpy as np

from bisect import bisect
from framework import file_m2k
from framework.post_proc import envelope
from parameter_estimation.intsurf_estimation import profile_fadmm, img_line_improved

# === Matplotlib Configuration ===
matplotlib.use('TkAgg')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.size": 10,
    "font.weight": "normal",
})
linewidth = 6.3091141732 # LaTeX linewidth

#%%

def surfaces(img, threshold=0.99, lamb=1e-2, rho=100):
    max = np.max(img, axis=0)
    img = img / max
    a = img_line_improved(img, threshold)
    z = a[0].astype(int)
    w = np.diag((a[1]))
    idx, resf, kf, pk, sk = profile_fadmm(w, z, lamb=lamb, x0=z, rho=rho,
                                          eta=.999, itmax=25, tol=1e-3)
    idx = idx.astype(int)
    return idx


root = "../data/grooves_sweep/"
positions = ['pos0.m2k', 'pos90.m2k', 'pos180.m2k', 'pos270.m2k']

data_ref = file_m2k.read(root + "ref.m2k", type_insp='contact', freq_transd=5, bw_transd=0.5, tp_transd='gaussian', sel_shots=0)

# Extrai o tempo e seta os angulos
t_span = data_ref.time_grid[:, 0]
ang_span = np.linspace(-45, 45, 181)

# Variaveis uteis
log_cte = 1e-6
c = 5813

t_span_croped = t_span

for i, pos in enumerate(positions):
    print(pos)
    data = file_m2k.read(root + pos, type_insp='contact', water_path=0, freq_transd=5, bw_transd=0.5,
                         tp_transd='gaussian', sel_shots=0)

    # Envelope e escala LOG
    if "ref" in pos:
        pass
    else:
        if i == 0:
            data_sum = np.sum(data.ascan_data - data_ref.ascan_data, axis=2)
            data_concatenated = data_sum[..., 0]
        else:
            data_sum = np.sum(data.ascan_data - data_ref.ascan_data, axis=2)
            data_concatenated = np.concatenate((data_concatenated, (data_sum[..., 0])), axis=1)

img = envelope(data_concatenated, axis=0)
img_log = np.log10(img + log_cte)
img_normalized = np.log10(img / img.max() + log_cte)

ang_span_full = np.linspace(0, 360, data_concatenated.shape[1])


# Cropando a superficie externa para passar o SEAM
t_end = bisect(t_span_croped, 57)
crop_externa = img_log[:t_end]

# Passando seam na EXTERNA
tau_ext_idx = surfaces(crop_externa, threshold=.995, lamb=1000, rho=10)  # Retorna os idx da superficie
tau_ext = t_span_croped[tau_ext_idx]  # Pega os tempos do SEAM para plot

# Cropando para a INTERNA para passar o SEAM
crop_interna = img_log[t_end::]

# Passando seam na INTERNA
tau_int_idx = surfaces(crop_interna / crop_interna.max(), threshold=.995, lamb=20,
                       rho=100) + t_end  # Retorna os idx da superficie
tau_int = t_span_croped[tau_int_idx]  # Pega os tempos do SEAM para plot

ext_time_mean = np.mean(tau_ext)

# Identifica a superfície externa no S-scan
ref_0 = int(np.mean(tau_ext))

# Ajusta os eixo z com o zero no centro do tubo
z_span_raw = c * (t_span_croped.max() - t_span_croped) * 1e-3 / 2
z_span = 70 + z_span_raw - z_span_raw[ref_0]

# Calcula as coordenadas x e z para superfície interna curva
bot_plain = z_span[tau_int_idx]
z_bot = bot_plain * np.cos(np.deg2rad(ang_span_full))
x_bot = bot_plain * np.sin(np.deg2rad(ang_span_full))

# Calcula as coordenadas x e z para superfície externa curva
top_plain = z_span[tau_ext_idx]
z_top = top_plain * np.cos(np.deg2rad(ang_span_full))
x_top = top_plain * np.sin(np.deg2rad(ang_span_full))

# Calcula as matrizes de coordenadas x e z para o plotar o S-scan corrigido
raios = np.linspace(z_span[0], z_span[-1], len(z_span))
ang_rad = np.deg2rad(ang_span_full)
X, R = np.meshgrid(ang_rad, raios)
X = X.reshape(X.shape[0] * X.shape[1])
R = R.reshape(R.shape[0] * R.shape[1])
x = (R * np.sin(X)).reshape([len(raios), len(ang_rad)])
z = (R * np.cos(X)).reshape([len(raios), len(ang_rad)])

#%%

fig, ax = plt.subplots(figsize=(linewidth * .6, linewidth * .5))
mappable = ax.pcolor(x, z, img_normalized, cmap='YlGnBu', vmin=-6, vmax=0)
plt.colorbar(mappable)
ax.plot(x_top, z_top, '-', color='k', linewidth=1.5, label='Outer Surface')
ax.plot(x_bot, z_bot, '-', color='k', linewidth=1.5, label='Inner Surface')
# ax.legend(loc='center', fancybox=True, frameon=False)
ax.set_aspect(1)
plt.xlim([-75, 75])
plt.ylim([-75, 75])
plt.xlabel(r"x-axis / ($\mathrm{mm}$)")
plt.ylabel(r"z-axis / ($\mathrm{mm}$)")
plt.xticks(np.linspace(-70, 70, 5))
plt.yticks(np.linspace(-70, 70, 5))
plt.grid(alpha=.25)
plt.tight_layout()
plt.savefig("../figures/grooves_tube_cscan.jpg")
plt.show()

#%%
angles_ranges = np.array([[10, 32], [62, 84], [103, 121], [152, 171], [190, 210], [243, 263], [280, 301], [331, 357], [36, 55]])
names = ['Médio Redondo 2', 'Médio redondo 1', 'Grande Redondo', 'Grande Flat', 'Médio Flat 2', 'Médio Flat 1', 'Pequeno Flat', 'Pequeno Redondo', 'Espessura do Tubo (ext até in)']

plt.figure(figsize=(linewidth * .5, 3))
plt.title("Tubulação Com Sulcos")
plt.imshow(img_normalized, extent=[ang_span_full[0], ang_span_full[-1], t_span_croped[-1], t_span_croped[0]], aspect='auto', interpolation="None",
           cmap='YlGnBu', vmin=-6, vmax=0)
plt.plot(ang_span_full, tau_int, 'r:', label="SEAM")
plt.plot(ang_span_full, tau_ext, 'r:')
plt.xlabel("Ângulo de varredura da tubulação")
plt.ylabel(r"Tempo em $\mathrm{\mu s}$")
plt.show()

#%%
# plt.figure()
# plt.title("Plot Plano 360º")
# plt.imshow(img_normalized, aspect='auto', cmap='magma',
#            extent=[ang_span_full[0], ang_span_full[-1], t_span_croped[-1], t_span_croped[0]],
#            vmin=-6, vmax=0)
# plt.ylabel(r"Tempo em $\mathrm{\mu s}$")
# plt.xlabel(r"Ângulo de varredura da tubulação")
# plt.show()

#%%
#
# # Plot do crop final
# plt.figure(figsize=(10, 5))
# plt.title("Tubulação Com Sulcos")
# plt.imshow(img_log, extent=[ang_span_full[0], ang_span_full[-1], t_span_croped[-1], t_span_croped[0]], aspect='auto',
#            cmap='RdBu')
# plt.xlabel("Ângulo de varredura da tubulação")
# plt.ylabel(r"Tempo em $\mathrm{\mu s}$")
# plt.plot(ang_span_full, tau_int, 'r:', label="SEAM")
# plt.plot(ang_span_full, tau_ext, 'r:')
#
# for i, rang in enumerate(angles_ranges):
#     a0 = bisect(ang_span_full, rang[0])
#     af = bisect(ang_span_full, rang[1])
#
#     tau_int_croped = tau_int[a0:af]
#     tau_ext_croped = tau_ext[a0:af]
#
#     speci_idx = np.unravel_index(np.argmin(tau_int_croped), shape=tau_int_croped.shape)
#     speci_time = tau_int_croped[speci_idx[0]]
#     outter_time = tau_ext_croped[speci_idx[0]]
#
#     remaining_tickeness = ((speci_time - outter_time) * 1e-3 * c) / 2
#
#     angle = ang_span_full[speci_idx[0] + a0]
#     plt.plot(angle, speci_time, 'kx')
#
#     print(f"{names[i]} = {remaining_tickeness:.2f}")
# plt.show()
###################################################################################################################################
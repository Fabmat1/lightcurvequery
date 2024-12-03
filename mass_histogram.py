from common_functions import *
import matplotlib.font_manager as fm
from matplotlib import rc
fm.findSystemFonts(fontpaths=None, fontext='ttf')

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

if __name__ == '__main__':
    title_fontsize  = 16
    label_fontsize  = 16
    legend_fontsize = 12
    tick_fontsize   = 14
    stars = load_solved_stars()

    primary_masses = np.array([star.m_1 for star in stars])
    primary_mass_errs = np.array([(star.m_1_err_lo, star.m_1_err_hi) for star in stars])
    secondary_masses = np.array([star.m_2 for star in stars])
    secondary_mass_errs = np.array([(star.m_2_err_lo, star.m_2_err_hi) for star in stars])

    mask = ~np.logical_or(np.logical_or(pd.isnull(primary_masses), pd.isnull(primary_mass_errs[:,0])), np.logical_or(pd.isnull(secondary_masses), pd.isnull(secondary_mass_errs[:,0])))
    primary_masses = primary_masses[mask]
    primary_mass_errs = primary_mass_errs[mask]
    secondary_masses = secondary_masses[mask]
    secondary_mass_errs = secondary_mass_errs[mask]

    theospace = np.linspace(0, 1.4587, 1000)
    plt.errorbar(primary_masses, secondary_masses, color="navy", xerr=primary_mass_errs.T, yerr=secondary_mass_errs.T, capsize=3, fmt='o', label='Data')
    plt.plot(theospace, np.flip(theospace), linestyle="--", color="darkred", label='Combined Chandrasekhar Mass Limit')
    # plt.title("Masses of Solved Systems", fontsize=title_fontsize)
    plt.xlabel(r"Primary Mass [$M_{\odot}$]", fontsize=label_fontsize)
    plt.ylabel(r"Companion Mass [$M_{\odot}$]", fontsize=label_fontsize)
    plt.xlim(0, np.max(primary_masses)*1.1)
    plt.ylim(0, np.max(secondary_masses)*1.1)
    plt.legend(loc="upper left", fontsize=legend_fontsize)
    plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    plt.tight_layout()
    plt.savefig("other_plots/mass_scatter.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()

    title_fontsize += 2
    label_fontsize += 2
    legend_fontsize += 2
    tick_fontsize += 2

    bin_edges = np.linspace(0, 1.2, 13)
    # Plot for primary masses
    plt.hist(primary_masses, bins=bin_edges, edgecolor='black', color='navy')
    # plt.title('Histogram of Primary Star Masses', fontsize=title_fontsize)
    plt.xlabel('Mass (Solar Masses)', fontsize=label_fontsize)
    plt.ylabel('Counts', fontsize=label_fontsize)
    plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    plt.xlim(0, 1.2)
    plt.tight_layout()
    plt.savefig("other_plots/primary_histogram.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()

    print(np.mean(primary_masses), np.median(primary_masses))
    print(np.std(primary_masses))

    # Plot for secondary masses
    plt.hist(secondary_masses, bins=bin_edges, edgecolor='black', color='darkred')
    # plt.title('Histogram of Secondary Star Masses', fontsize=title_fontsize)
    plt.xlabel('Mass (Solar Masses)', fontsize=label_fontsize)
    plt.ylabel('Counts', fontsize=label_fontsize)
    plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    plt.xlim(0, 1.2)
    plt.tight_layout()
    plt.savefig("other_plots/secondary_histogram.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()
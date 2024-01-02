import numpy as np

class Life:

    earth_gravity_log10 = 2.992 # log10(cm/s**2)
    earch_equilibrium_temperature = 255.0 # K
    percent_unknown = .25 # percent of planets where we don't know if life exists

    def __init__(
            self, 
            planet_radius_vs_earth, 
            planet_mass_vs_earth, 
            planet_equilibrium_temperature, 
            stellar_surface_gravity
        ):
        self.planet_radius_vs_earth = planet_radius_vs_earth
        self.planet_mass_vs_earth = planet_mass_vs_earth
        self.planet_equilibrium_temperature = planet_equilibrium_temperature
        self.stellar_surface_gravity = stellar_surface_gravity

    def does_life_exist(self):
        life_possible = self._is_life_possible()
        life_likely = self._is_life_likely()
        if not life_possible:
            life_exists = False
        else:
            if life_likely:
                if np.random.random() < .9:
                    life_exists = True
                else:
                    life_exists = False
            else:
                if np.random.random() < .6:
                    life_exists = True
                else:
                    life_exists = False
        return {
            'life_exists': life_exists,
            'life_possible': life_possible,
            'life_likely': life_likely
        }

    def _is_life_possible(self):
        possible_due_to_temp = abs(self.planet_equilibrium_temperature - self.earch_equilibrium_temperature) < 600
        possible_due_to_gravity = abs(self.stellar_surface_gravity - self.earth_gravity_log10) < 6
        if possible_due_to_temp and possible_due_to_gravity:
            return True
        else:
            return False

    def _is_life_likely(self):
        likely_due_to_temp = abs(self.planet_equilibrium_temperature - self.earch_equilibrium_temperature) < 400
        likely_due_to_gravity = abs(self.stellar_surface_gravity - self.earth_gravity_log10) < 5
        likely_due_to_mass = abs(self.planet_mass_vs_earth - 1) < .7
        likely_due_to_radius = abs(self.planet_radius_vs_earth - 1) < .7
        if likely_due_to_temp and likely_due_to_gravity and likely_due_to_mass and likely_due_to_radius:
            return True
        else:
            return False
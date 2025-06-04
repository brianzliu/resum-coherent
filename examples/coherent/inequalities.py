from resum.multi_fidelity_gaussian_process import MFGPInequalityConstraints
import numpy as np

class InequalityConstraints(MFGPInequalityConstraints):
    def __init__(self):
        # Initialize any helper functions if needed
        pass

    def evaluate(self, x):
        """
        Evaluate inequality constraints for coherent neutrino scattering detector design.
        
        Parameters in x (theta_headers - actual experimental design parameters):
        - x[:, 0]: water_shielding_mm (water shielding thickness in mm)
        - x[:, 1]: veto_thickness_mm (veto detector thickness in mm)
        """
        super().evaluate(x)
        delta_x = np.ones(len(x))
        
        for i, xi in enumerate(x[:, :-1]):
            # Design constraints for coherent neutrino detector optimization
            
            water_shielding = xi[0]  # mm
            veto_thickness = xi[1]   # mm
            
            # 1. Physical bounds based on actual experimental range
            # Water shielding: 16.2mm (Veto4) to 116.2mm (Veto1)
            if water_shielding < 10.0 or water_shielding > 120.0:
                delta_x[i] = 0.0
                continue
            
            # 2. Veto thickness bounds based on experimental range
            # Veto thickness: 10mm (Veto1) to 110mm (Veto4)
            if veto_thickness < 5.0 or veto_thickness > 115.0:
                delta_x[i] = 0.0
                continue
            
            # 3. Physical trade-off constraints
            # More water shielding generally allows for thinner veto (and vice versa)
            # This reflects the experimental design logic
            total_shielding = water_shielding + veto_thickness
            
            # Reasonable total shielding range (based on experimental configurations)
            if total_shielding < 50.0:  # Too little total shielding
                delta_x[i] = 0.0
                continue
                
            if total_shielding > 150.0:  # Too much total shielding (cost/space)
                delta_x[i] = 0.0
                continue
            
            # 4. Minimum effectiveness constraints
            # Need some minimum shielding for both components
            if water_shielding < 15.0 and veto_thickness < 15.0:  # Both too thin
                delta_x[i] = 0.0
                continue
            
            # 5. Practical engineering constraints
            # Avoid extreme ratios that might be impractical
            shielding_ratio = water_shielding / veto_thickness
            if shielding_ratio > 10.0 or shielding_ratio < 0.1:  # Extreme ratios
                delta_x[i] = 0.0
                continue
            
            # If all constraints are satisfied
            delta_x[i] = 1.0
            
        return delta_x[:, None] 
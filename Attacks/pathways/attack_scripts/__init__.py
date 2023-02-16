from .base_attacks import base_adv

def Adv_Attack(attack_type):

    if attack_type in ['fgsm', 'rfgsm', 'pgd', 'mifgsm', 'dim', 'pifgsm']:
        return base_adv
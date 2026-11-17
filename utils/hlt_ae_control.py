"""Opt-in event-random AE control; legacy splitter remains unchanged."""
import numpy as np


ROLES = ('train', 'validation', 'calibration', 'check')


def random_category_split(categories, seed=42, fractions=(.9,.05,.025,.025)):
    categories=np.asarray(categories)
    fractions=np.asarray(fractions,dtype=float)
    if categories.ndim!=1 or len(fractions)!=4 or np.any(fractions<=0) or not np.isclose(fractions.sum(),1):
        raise ValueError('Require a category vector and four positive fractions summing to one')
    rng=np.random.default_rng(seed)
    parts={role:[] for role in ROLES}
    for category in np.unique(categories):
        rows=np.flatnonzero(categories==category)
        if len(rows)<4:
            raise ValueError(f'Category {category} has fewer than four events')
        exact=len(rows)*fractions
        counts=np.maximum(1,np.floor(exact).astype(int))
        while counts.sum()>len(rows):
            eligible=np.where(counts>1,counts-exact,-np.inf)
            counts[np.argmax(eligible)]-=1
        while counts.sum()<len(rows):
            counts[np.argmax(exact-counts)]+=1
        shuffled=rng.permutation(rows)
        for role, subset in zip(ROLES,np.split(shuffled,np.cumsum(counts)[:-1])):
            parts[role].append(subset)
    return {role:rng.permutation(np.concatenate(parts[role])).astype(np.int64) for role in ROLES}


def validate_roles(roles, size):
    indices={role:np.asarray(roles[role],dtype=np.int64) for role in ROLES}
    if any(a.ndim!=1 or not len(a) for a in indices.values()):
        raise ValueError('Every role must be a nonempty vector')
    combined=np.concatenate(list(indices.values()))
    if len(combined)!=size or not np.array_equal(np.sort(combined),np.arange(size)):
        raise ValueError('AE roles must partition the sample without overlap or missing rows')
    return indices

"""BeliefModel: where the target probably is, updated after each miss.

Not part of the paper (which plans once); needed because object search
observes and replans."""


class BeliefModel:
    """Prior and posterior probability per space (GUID -> p, summing to 1).

    Miss update (Bayes with detector miss rate m = P(not detected | present)):
        P(r) <- P(r) * m for the searched room r, then renormalise all rooms.
    A room is never set to 0 and the ranking model is not asked again."""

    def __init__(self):
        self.prior = {}
        self._posterior = {}

    @property
    def has_prior(self):
        return bool(self._posterior)

    def set_prior(self, probabilities):
        total = sum(probabilities.values())
        if total <= 0:
            raise ValueError("prior probabilities sum to 0")
        self.prior = {s: p / total for s, p in probabilities.items()}
        self._posterior = dict(self.prior)

    def update_on_miss(self, space, miss_rate):
        if not 0.0 <= miss_rate <= 1.0:
            raise ValueError(f"miss_rate must be in [0, 1], got {miss_rate}")
        if space not in self._posterior:
            return
        post = dict(self._posterior)
        post[space] *= miss_rate
        total = sum(post.values())
        if total > 0:
            self._posterior = {s: p / total for s, p in post.items()}

    def posterior(self):
        return dict(self._posterior)

    def eligible(self, visits, max_visits, allowed=None):
        """Spaces with fewer than max_visits visits (and in `allowed`, if given)."""
        return [s for s in self._posterior
                if visits.get(s, 0) < max_visits and (allowed is None or s in allowed)]

    def mass(self, spaces):
        return sum(self._posterior.get(s, 0.0) for s in spaces)

    def should_stop(self, eligible, residual_mass, unreachable=()):
        """(stop?, reason). Stop when no space is eligible or the eligible
        (reachable, not yet searched) spaces together hold less than
        residual_mass of the (un-normalised) posterior. Assumes the target is
        somewhere in the building. The reason always states the mass in the
        unreachable spaces, so a "remaining probability" never hides it."""
        rest = f"; unreachable rooms hold {self.mass(unreachable):.3f}"
        if not eligible:
            return True, "no reachable room left to search" + rest
        mass = self.mass(eligible)
        if mass < residual_mass:
            return True, f"remaining probability in reachable rooms {mass:.3f} < residual_mass {residual_mass:.3f}" + rest
        return False, ""

    def to_state(self):
        return {"prior": dict(self.prior), "posterior": dict(self._posterior)}

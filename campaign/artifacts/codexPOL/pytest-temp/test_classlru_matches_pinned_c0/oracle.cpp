#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>
#define GGML_ASSERT(x) do { if (!(x)) std::abort(); } while (0)
#define GGML_ABORT(...) std::abort()
struct moe_cache_lru {
    static constexpr int   n_class  = 8; // floor(log2(1/(1 - 2^(-1/crf_half_life)))) = 7, class of CRF_max for half-life of 128
    static constexpr float crf_half_life = 128.0f; // tokens

    int32_t n_expert = 0;
    int32_t n_slots  = 0;
    int32_t empty    = 0; // slot of the experts that are not cached

    // [n_layer] data of the slot maps, the experts at slot n_slots or later are not cached
    // the experts computed on the CPU read zero slots after the cached ones on the device
    std::vector<int32_t *> slot_map;
    std::vector<int32_t>   key_of;   // [n_slots] key of the expert in the slot, -1 if empty

    // one LRU list per class, the head is the least recently used end
    std::vector<int32_t> prev;    // [n_slots] -1 at the ends
    std::vector<int32_t> next;    // [n_slots]
    int32_t head[n_class];
    int32_t tail[n_class];
    std::vector<int8_t>  list_of; // [n_slots] class list of the slot
    int32_t n_filled = 0;         // slots that hold an expert

    std::vector<float>   crf;      // [n_layer*n_expert]
    std::vector<int64_t> last_use; // [n_layer*n_expert] layer clock at the last use, -1 if not used
    std::vector<int64_t> clock;    // [n_layer] tokens seen by the layer (as added by plan() for each ubatch)

    std::vector<uint32_t> seen; // [n_expert] seen[e] == seen_gen if expert e is selected by the current plan()
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;  // the distinct selected experts
    std::vector<int32_t> misses;
    std::vector<uint64_t> token_zero; // [n_tokens] zero slots used by each token

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        GGML_ASSERT(n_class == (int) std::log2(1.0f/(1.0f - std::exp2(-1.0f/crf_half_life))) + 1);
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        this->empty    = n_slots;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        for (int c = 0; c < n_class; ++c) {
            head[c] = -1;
            tail[c] = -1;
        }
        head[0] = 0;
        tail[0] = n_slots - 1;
        list_of.assign(n_slots, 0);
        crf.assign((size_t) n_layer*n_expert, 0.0f);
        last_use.assign((size_t) n_layer*n_expert, -1);
        clock.assign(n_layer, 0);
        seen.assign(n_expert, 0);
    }

    // move slot s to the most recently used end of list c
    void move(int32_t s, int c) {
        const int from = list_of[s];
        if (from == c && s == tail[from]) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head[from] = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            tail[from] = prev[s];
        }

        prev[s] = tail[c];
        next[s] = -1;
        if (tail[c] >= 0) {
            next[tail[c]] = s;
        } else {
            head[c] = s;
        }
        tail[c] = s;
        list_of[s] = c;
    }

    // key = index of an expert of a layer in crf and last_use
    int32_t make_key(int32_t il, int32_t e) const { return il*n_expert + e; }
    int32_t key_layer (int32_t key) const { return key / n_expert; }
    int32_t key_expert(int32_t key) const { return key % n_expert; }

    // for a key after use(): its CRF is >= 1, so log2 is >= 0 and the cast is the floor
    int class_of(int32_t key) const {
        return std::min(n_class - 1, (int) std::log2(crf[key]));
    }

    // the CRF of a key, decayed by the time since its last use
    float decayed_crf(int32_t il, int32_t key) const {
        return crf[key]*std::exp2(-(clock[il] - last_use[key])/crf_half_life);
    }

    // an unused key gets 1 because crf[key] is 0 prior to first use
    void use(int32_t il, int32_t key) {
        crf[key]      = decayed_crf(il, key) + 1.0f;
        last_use[key] = clock[il];
    }

    // the experts selected by this plan() must not be evicted, nor the slots this plan() filled already
    bool is_selected(int32_t il, int32_t key) const {
        if (key < 0) {
            return false; // unused slot
        }
        if (key_layer(key) != il) {
            return false; // seen is per expert index, an expert of another layer can have the same index
        }
        return seen[key_expert(key)] == seen_gen;
    }

    // the least recently used slot of list c that this plan() does not select, -1 if none
    int32_t candidate(int c, int32_t il) const {
        for (int32_t s = head[c]; s >= 0; s = next[s]) {
            if (!is_selected(il, key_of[s])) {
                return s;
            }
        }
        return -1;
    }

    int32_t victim(int32_t il) const {
        int32_t res = -1;
        double res_score = -1.0;
        for (int c = 0; c < n_class; ++c) {
            const int32_t s = candidate(c, il);
            if (s < 0) {
                continue;
            }
            const int32_t key = key_of[s];
            if (key < 0) {
                return s; // unused slot
            }
            const int64_t age   = clock[key_layer(key)] - last_use[key];
            const double  score = std::ldexp((double) age, -c); // age/2^class
            if (score > res_score) {
                res       = s;
                res_score = score;
            }
        }
        if (res < 0) {
            GGML_ABORT("no MoE cache victim");
        }
        return res;
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    // give a slot to each expert selected by ids (n_tokens tokens) in layer il
    // n_cpu(n) of the n misses are left to the CPU and read a zero slot on the device, they are not cached
    // a zero slot is used at most once per token, a miss that does not get one is cached instead
    // returns false if the ids select more distinct experts than there are slots
    template <typename F>
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, int64_t n_tokens, int32_t n_used, int32_t n_zero, F && n_cpu,
            std::vector<fill> & fills, std::vector<fill> & cpu, size_t & n_hit) {
        fills.clear();
        cpu.clear();
        n_hit = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_slots) {
            return false;
        }

        int32_t * slots = slot_map[il];

        // update the CRF of all selected experts and promote hits before choosing victims for misses
        misses.clear();
        for (int32_t e : uniq) {
            const int32_t key = make_key(il, e);
            use(il, key);
            if (slots[e] < n_slots) {
                move(slots[e], class_of(key));
                n_hit++;
            } else {
                misses.push_back(e);
            }
        }

        // the misses with the lowest CRF are left to the CPU, a miss that keeps being used gains CRF and enters the cache
        // all misses are used at the same clock, so their CRF compare without decay
        const size_t n_c = std::min(n_cpu(misses.size()), misses.size());
        if (n_c > 0) {
            std::sort(misses.begin(), misses.end(), [&](int32_t a, int32_t b) {
                const float crf_a = crf[make_key(il, a)];
                const float crf_b = crf[make_key(il, b)];
                return crf_a != crf_b ? crf_a < crf_b : a < b;
            });
            GGML_ASSERT((size_t) n_tokens*n_used == n_ids);
            token_zero.assign(n_tokens, 0);
            size_t n_keep = n_c;
            for (size_t i = 0; i < n_keep; ) {
                const int32_t e = misses[i];
                uint64_t used = 0;
                for (int64_t it = 0; it < n_tokens; ++it) {
                    for (int32_t k = 0; k < n_used; ++k) {
                        if (ids[it*n_used + k] == e) {
                            used |= token_zero[it];
                        }
                    }
                }
                int32_t z = 0;
                while (z < 64 && (used >> z & 1)) {
                    z++;
                }
                if (z >= n_zero) {
                    std::swap(misses[i], misses[--n_keep]);
                    continue;
                }
                for (int64_t it = 0; it < n_tokens; ++it) {
                    for (int32_t k = 0; k < n_used; ++k) {
                        if (ids[it*n_used + k] == e) {
                            token_zero[it] |= 1ull << z;
                        }
                    }
                }
                slots[e] = n_slots + z;
                cpu.push_back({ e, n_slots + z });
                i++;
            }
            misses.erase(misses.begin(), misses.begin() + n_keep);
        }

        // fill the misses in expert order: victim() takes unused slots, and slots filled by one plan(), in list order,
        // so consecutive experts often get consecutive slots and one upload
        std::sort(misses.begin(), misses.end());
        for (int32_t e : misses) {
            const int32_t s = victim(il);
            if (key_of[s] >= 0) {
                slot_map[key_layer(key_of[s])][key_expert(key_of[s])] = empty;
            } else {
                n_filled++;
            }
            const int32_t key = make_key(il, e);
            key_of[s] = key;
            slots[e]  = s;
            move(s, class_of(key));
            fills.push_back({ e, s });
        }
        clock[il] += n_tokens;
        return true;
    }
};


int main() {
    int cap, q; std::cin >> cap >> q;
    moe_cache_lru cache; cache.init(4, 128, cap);
    std::vector<std::vector<int32_t>> maps(4, std::vector<int32_t>(128, cap));
    std::vector<uint32_t> carry(4, 0);
    for (int i=0;i<4;i++) cache.slot_map[i] = maps[i].data();
    int layer;
    while (std::cin >> layer) {
        int32_t ids[8]; for (auto &e : ids) std::cin >> e;
        std::vector<moe_cache_lru::fill> fills, cpu;
        size_t hits;
        uint32_t share = cache.n_filled == cap ? q : 0;
        auto ncpu = [&](size_t n) {
            uint64_t acc = n*uint64_t(share) + carry[layer];
            carry[layer] = acc & 65535; return size_t(acc >> 16);
        };
        cache.plan(layer, ids, 8, 1, 8, q ? 8 : 0, ncpu, fills, cpu, hits);
        std::cout << hits << ' ' << fills.size() << ' ' << cpu.size();
        for (int s=0;s<cap;s++) std::cout << ' ' << cache.key_of[s] << ' ' << int(cache.list_of[s]);
        std::cout << '\n';
    }
}

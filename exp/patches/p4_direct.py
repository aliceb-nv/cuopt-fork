# Fresh staging in view+slot mode: pack straight from the node's vectors (no host-slab copy).
import sys
root=sys.argv[1]
p=root+'/cpp/src/dual_simplex/gpu_phase2/gpu_phase2.cu'; s=open(p).read()
a="""      if (command == gpu_batch_seat_command::FRESH_NODE) {
        const auto copy_seat = [seat](auto& slab, size_t width, const auto& source) {"""
b="""      // Fresh seats in slot mode upload straight from the node's vectors; the slab rows are
      // rewritten by the next harvest, and nothing reads them in between.
      if (command == gpu_batch_seat_command::FRESH_NODE && H->use_slots) {
        auto& F        = *H->fixed;
        const auto put = [&](auto& dev, const auto& source, size_t width) {
          if (width != 0) {
            H->pack.add(dev.data() + seat * width, source.data(), width * sizeof(source[0]), H->stream);
          }
        };
        put(F.x, item.sol->x, n);
        put(F.vstatus, *item.vstatus, n);
        put(F.basic_list, *item.basic_list, m);
        put(F.nonbasic_list, *item.nonbasic_list, n - m);
        put(F.y, item.sol->y, m);
        put(F.z, item.sol->z, n);
        put(F.dy_se, *item.dy_se, n);
        split(2);
      } else {
      if (command == gpu_batch_seat_command::FRESH_NODE) {
        const auto copy_seat = [seat](auto& slab, size_t width, const auto& source) {"""
assert s.count(a)==1; s=s.replace(a,b)
a="""        if (command == gpu_batch_seat_command::FRESH_NODE) {
          slice(F.y, H->h_y, m);
          slice(F.z, H->h_z, n);
          slice(F.dy_se, H->h_dy_se, n);
        }
      }
    } else {"""
b="""        if (command == gpu_batch_seat_command::FRESH_NODE) {
          slice(F.y, H->h_y, m);
          slice(F.z, H->h_z, n);
          slice(F.dy_se, H->h_dy_se, n);
        }
      }
      }
    } else {"""
assert s.count(a)==1; s=s.replace(a,b)
open(p,'w').write(s); print('ok')

#!/usr/bin/env python3
from utils_io import *
from utils_args import *
import yaml


def check_record (r):
  def chk (k, t):
    if k not in r or not isinstance (r[k], t):
      raise ValueError (f'missing or wrong `{k}\' entry in record')
  chk ('adversarials', list)
  chk ('passed_tests', list)
  chk ('norm', str)


def read_yaml_record_file (f):
  with open (f, 'r') as f:
    r = yaml.safe_load (f)
  try:
    check_record (r)
    return r
  except ValueError as e:
    raise ValueError (f'Malformed record file: {e}')


class Record:
  
  def __init__(self, **kwds):
    super().__init__(**kwds)
    self.roots = ()
    self.nodes = {}


  def reset_from_yaml_record (self, record, dir = None):
    self.roots = ()
    self.nodes = {}

    lr = [None] * (len (record['passed_tests']) +
                   len (record['adversarials']))

    for t in record['passed_tests']:
      lr[t['index']] = t
      t['status'] = 'pass' if 'gen_test_id' in t else 'raw'

    for t in record['adversarials']:
      lr[t['index']] = t
      t['status'] = 'adversarial'

    def tnode_idx (t):
      return -t['index'] - 1 if 'gen_test_id' not in t else t['gen_test_id']

    # Note some elements in [lr] may still be [None] if there are
    # duplicated inputs in the initial test suite.  We can safely
    # ignore those indexes as no generated test case may derive from
    # them (i.e. those point to the index of the unique duplicated
    # input that is in [lr]).

    for t in lr:
      if t is None: continue
      id = tnode_idx (t)
      t['id'] = id
      t['childs'] = ()
      self.nodes[id] = t
      if 'origin_index' in t:
        origin = self.nodes[tnode_idx (lr[t['origin_index']])]
        t['origin'] = origin['id']
        origin['childs'] += (id,)
      else:
        t['origin'] = None
        self.roots += (id,)

    def set_image (t, f):
      if os.path.exists (os.path.join (dir, f) if dir else f):
        t['image'] = f

    for t in lr:
      if t is None: continue
      if t['status'] == 'raw':
        tchilds = t["childs"]
        if tchilds != ():
          set_image (t, f'{self.nodes[tchilds[0]]["id"]}-original-{t["label"]}.png')
      elif t['status'] == 'pass':
        set_image (t, f'{t["id"]}-ok-{t["label"]}.png')
      elif t['status'] == 'adversarial':
        set_image (t, f'{t["id"]}-adv-{t["label"]}.png')

    
  @classmethod
  def from_yaml_record (cls, yr, **_):
    self = cls.__new__(cls)
    self.reset_from_yaml_record (yr, **_)
    return self


  def traverse (self, exclude_dangling_roots = True):
    """Parents first; yields pairs of dictionaries `(node, parent)`, with
    `parent = None` for roots."""

    def node (t):
      parent = self.nodes[t['origin']] if t['origin'] is not None else None
      yield (t, parent)
      for c in t['childs']:
        yield from node (self.nodes[c])

    for ridx in self.roots:
      r = self.nodes[ridx]
      if not exclude_dangling_roots or r['childs'] != ():
        for x in node (r): yield x


# ---


try:
  from pyvis.network import Network
  _has_pyvis = True
  _pyvis_all_images = ('raw', 'pass', 'adversarial')

  def record_to_pyvis (record,
                       show_images = _pyvis_all_images,
                       level_scaling =  True,
                       **_):
    n = Network (directed = True, **_)

    mean_dist = 1.
    if level_scaling:
      mean_dist = 0.0
      for i, (t, _) in enumerate (record.traverse (exclude_dangling_roots = True)):
        if 'origin_dist' in t:
          mean_dist += t['origin_dist'] / i
      mean_dist /= 16.

    for t, parent in record.traverse (exclude_dangling_roots = True):
      tid = t['id']
      label = t['label']
      props = dict ()
      
      if t['status'] == 'raw':
        props['label'] = 'raw'
        props['title'] = f'Label: {label}'
      elif t['status'] == 'pass':
        props['color'] = 'green'
        props['title'] = f'Predicted label: {label}'
      elif t['status'] == 'adversarial':
        props['color'] = 'red'
        props['title'] = f'Predicted label: {label}'
        
      if 'image' in t and t['status'] in show_images:
        props['image'] = t['image']
        props['shape'] = 'image'
        props['size'] = 40
        props['shadow'] = True

      if 'root_dist' in t:
        dist = t['root_dist']
        props['title'] += f'<br/>Distance to root: {dist:.4g}'
        props['level'] = dist if dist <= 0 or dist >= 1 else dist / mean_dist
      else:
        props['level'] = 0

      n.add_node (tid, **props)

      if parent is not None:
        dist = t['origin_dist']
        n.add_edge (parent['id'], tid,
                    label = f'{dist:.3}',
                    color = props['color'],
                    width = 3)

    return n

except:
  _has_pyvis = False


# ---

ap = argparse.ArgumentParser \
  (description = 'Interactive visualisation of DeepConcolic testing record',
   formatter_class = argparse.ArgumentDefaultsHelpFormatter)
add_named_n_pos_args (ap, 'record', ('--record', '-r'), metavar = 'YAML',
                      help = 'record file')
if _has_pyvis:
  _pyvis_all_buttons = ('nodes', 'edges', 'layout', 'interaction',
                        'manipulation', 'physics', 'selection', 'renderer')
  _pyvis_default_buttons = ('layout', 'physics')
  try:
    import json                 # for dict load
    ap.add_argument ('--pyvis-open', '-pvopen', action = 'store_true',
                     help = 'open the saved HTML network')
    ap.add_argument ('--pyvis-non-hierarchical', '-pvnh', action = 'store_true',
                     help = 'do not use a hierarchical layout to render '
                     'the pyvis network')
    ap.add_argument ('--pyvis-network-options', '-pvo', type = json.loads,
                     default = '{"width": "98vw", "height": "98vh"}',
                     help = 'dictionary of options for pyvis network creation')
    ap.add_argument ('--pyvis-show-images', '-pvimgs', nargs = '+',
                     choices = _pyvis_all_images, default = _pyvis_all_images,
                     help = 'images to show in the pyvis network visualisation')
    ap.add_argument ('--pyvis-show-buttons', '-pvbuttons', nargs = '*',
                     choices = _pyvis_all_buttons,
                     default = _pyvis_default_buttons,
                     help = 'buttons to show in the resulting HTML page')
  except: pass

def get_args (args = None, parser = ap):
  return parser.parse_args () if args is None else args

def main (args = None, parser = ap):
  try:
    args = get_args (args, parser = parser)
    destdir = os.path.dirname (args.record)
    record = Record.from_yaml_record (read_yaml_record_file (args.record),
                                      dir = destdir)
    if _has_pyvis:
      layout = 'hierarchical' if not args.pyvis_non_hierarchical else None
      n = record_to_pyvis (record, layout = layout,
                           show_images = args.pyvis_show_images,
                           **args.pyvis_network_options)

      if not args.pyvis_non_hierarchical:
        # A tad on the brittle side:
        n.options.layout.hierarchical.direction = "LR"

      if len (args.pyvis_show_buttons) != 0:
        n.show_buttons (filter_ = args.pyvis_show_buttons)

      htmldest = f'{os.path.splitext (args.record)[0]}.html'
      xmsg, save = ((', and opening', n.show) if args.pyvis_open else \
                    ('', n.save_graph))
      p1 (f'Saving pyvis record graph into `{htmldest}\'{xmsg}')
      save (htmldest)
    else:
      sys.exit ('No backend available')
  except ValueError as e:
    sys.exit (f'Error: {e}')
  except FileNotFoundError as e:
    sys.exit (f'Error: {e}')
  except KeyboardInterrupt:
    sys.exit ('Interrupted.')

# ---

if __name__=="__main__":
  main ()
